import dgl
import torch.nn.functional as F
import torch
from model import SimaModel, MLPPredictor_simple
import numpy as np
import scipy.sparse as sp
import random
import networkx as nx
from multiprocessing import cpu_count
from tools import run_multithread, run_starmie_multithread, metrics, fabricated_to_source_filename
import itertools
from torchmetrics.functional import f1_score


############ Negative Sampling Strategies ############

def negative_sampling_all(graph):
    """
        Include all negative edges for training

        Input: a relatedness graph
        Output: the corresponding graph with all possible negative edges
    """
    src, dest = graph.edges()

    positive_no = graph.number_of_edges()

    adjacency_matrix = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dest.numpy())))
    adjacency_inverse = 1 - adjacency_matrix.todense() - np.eye(graph.number_of_nodes())

    neg_src, neg_dest = np.where(adjacency_inverse != 0)

    graph_neg = dgl.graph((neg_src.tolist(), neg_dest.tolist()), num_nodes=graph.number_of_nodes())

    negative_no = graph_neg.number_of_edges()

    ratio = int(negative_no/positive_no)

    return graph_neg


def negative_sampling_1(graph):
    """
        Negative sampling strategy #1 from the paper: randomly sample as many
        negative edges as the positive ones in the relatedness graph.

        Input: a relatedness graph
        Output: the corresponding graph with the sample of negative edges (undirected)
    """
    src, dest = graph.edges()

    sample_size = graph.number_of_edges()

    adjacency_matrix = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dest.numpy())))
    adjacency_inverse = 1 - adjacency_matrix.todense() - np.eye(graph.number_of_nodes())

    neg_src, neg_dest = np.where(adjacency_inverse != 0)

    neg_eids = np.random.choice(len(neg_src), sample_size)

    sample_neg_src, sample_neg_dest = neg_src[neg_eids], neg_dest[neg_eids]

    graph_neg = dgl.graph((sample_neg_src.tolist(), sample_neg_dest.tolist()), num_nodes=graph.number_of_nodes())

    return graph_neg


def negative_sampling_2(graph):
    """
        Negative sampling strategy #2 from the paper: randomly sample negative edges per node
        in the relatedness graph. The number of edges sampled are the same as the in-degree of the node.

        Input: a relatedness graph
        Output: the corresponding graph with the sample of negative edges (directed)
    """
    src, dest = graph.edges()

    adjacency_matrix = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dest.numpy())))
    adjacency_inverse = 1 - adjacency_matrix.todense() - np.eye(graph.number_of_nodes())

    neg_src, neg_dest = np.where(adjacency_inverse != 0)

    nodes_edges = {i: [] for i in range(graph.number_of_nodes())}

    for i in range(len(neg_src)):
        nodes_edges[neg_src[i]].append((neg_src[i], neg_dest[i]))

    edges_neg = []

    for k, v in nodes_edges.items():
        sample = random.sample(v, min(graph.in_degrees(k), len(v)))
        for s in sample:

            edges_neg.append((s[1], s[0]))
            nodes_edges[s[1]].remove((s[1], s[0]))

    edges_neg = torch.tensor(edges_neg, dtype=torch.long).t().contiguous()
    graph_neg = dgl.graph((edges_neg[0], edges_neg[1]))

    return graph_neg


def negative_sampling_3(graph, balance=False):
    """
            Negative sampling strategy #3 from the paper: randomly sample negative edges per node
            in the relatedness graph. Each node receives one negative sample per connected component
            in the original relatedness graph, i.e., receives negative samples for each other column domain.

            Input: a relatedness graph, balance parameter if we want to limit negative edges per node to the
            in-degree of each node.
            Output: the corresponding graph with the sample of negative edges (directed)
    """
    edges_neg = []

    graph_nx = graph.to_networkx().to_undirected()

    components = []

    for cc in nx.connected_components(graph_nx):
        components.append(list(cc))

    for i, cc in enumerate(components):
        for col1 in cc:
            edges = []
            for j, c in enumerate(components):
                if i != j:
                    sample_size = graph.in_degrees(col1)
                    cols = random.sample(c, min(sample_size, len(c)))
                    for col2 in cols:
                        edges.append((col2, col1))
            if graph.in_degrees(col1) < len(edges) and balance:

                edges_neg.extend(random.sample(edges, graph.in_degrees(col1)))
            else:
                edges_neg.extend(edges)

    edges_neg = torch.tensor(edges_neg, dtype=torch.long).t().contiguous()
    graph_neg = dgl.graph((edges_neg[0], edges_neg[1]))

    return graph_neg


def compute_loss(pos_score, neg_score, pos_weight):
    """
        Compute binary cross entropy loss based on predictions on the positive and negative
        edge samples.

        Input:
            pos_score: scores on positive edges
            neg_score: scores on negative edges
            pos_weight: weight of positive samples used in loss function
    """
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    scores_classes = (torch.sigmoid(scores) >= 0.5).int()
    f1 = f1_score(scores_classes, labels.int(), num_classes=2)
    weight_positives = torch.FloatTensor([pos_weight])
    return F.binary_cross_entropy_with_logits(scores, labels, pos_weight=weight_positives), f1


def train_model(graphs, graphs_neg, epochs, embed_size, incremental=False, rate=0.01, wd=0):
    """
        Train Casanova model based on the relatedness graphs and their negative counterparts.

        Input:
            graphs: list of relatedness graphs
            graphs_neg: list of relatedness graphs with negative edges
            epochs: number of epochs for training the model
            embed_size: size of output embeddings from GraphSAGE
            incremental: boolean parameter to control incremental training
            rate: learning rate
            wd: weight decay
        Output:
            model: trained model - including gnn and predictor
    """

    model = SimaModel(graphs[0].ndata['feat'].shape[1], embed_size, rate, wd)

    if incremental:
        for i in range(len(graphs)):
            graph_all = dgl.batch([graphs[j] for j in range(i + 1)])
            graph_neg_all = dgl.batch([graphs_neg[j] for j in range(i + 1)])
            for e in range(epochs):
                # forward
                h = model.gnn(graph_all, graph_all.ndata['feat'])
                pos_score = model.predictor(graph_all, h)
                neg_score = model.predictor(graph_neg_all, h)

                positive_weight = int(graph_neg_all.number_of_edges()/graph_all.number_of_edges())

                loss, f1 = compute_loss(pos_score, neg_score, positive_weight)

                # backward
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()

                if e % 10 == 0:
                    print('In epoch {}, loss {}, f1 {}'.format(e, loss, f1))
    else:  # train on all relatedness graphs at once
        # merge all relatedness graphs and their negative counterparts
        graph_all = dgl.batch([graphs[i] for i in range(len(graphs))])
        graph_neg_all = dgl.batch([graphs_neg[i] for i in range(len(graphs))])

        for e in range(epochs):
            # forward
            h = model.gnn(graph_all, graph_all.ndata['feat'])
            pos_score = model.predictor(graph_all, h)
            neg_score = model.predictor(graph_neg_all, h)
            positive_weight = int(graph_neg_all.number_of_edges() / graph_all.number_of_edges())

            loss, f1 = compute_loss(pos_score, neg_score, positive_weight)

            # backward
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            if e % 10 == 0:

                print('In epoch {}, loss {}, f1 {}'.format(e, loss, f1))

    return model

def train_mlp_model(graphs, graphs_neg, epochs, embed_size, incremental=False, rate=0.01, wd=0):

    model = MLPPredictor_simple(graphs[0].ndata['feat'].shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=wd)

    if incremental:
        for i in range(len(graphs)):
            graph_all = dgl.batch([graphs[j] for j in range(i + 1)])
            graph_neg_all = dgl.batch([graphs_neg[j] for j in range(i + 1)])
            for e in range(epochs):
                # forward
                pos_score = model(graph_all)
                neg_score = model(graph_neg_all)

                positive_weight = int(graph_neg_all.number_of_edges()/graph_all.number_of_edges())

                loss, f1 = compute_loss(pos_score, neg_score, positive_weight)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if e % 10 == 0:
                    print('In epoch {}, loss {}, f1 {}'.format(e, loss, f1))
    else:  # train on all relatedness graphs at once
            # merge all relatedness graphs and their negative counterparts
        graph_all = dgl.batch([graphs[i] for i in range(len(graphs))])
        graph_neg_all = dgl.batch([graphs_neg[i] for i in range(len(graphs))])

        for e in range(epochs):
            # forward
            h = model.gnn(graph_all, graph_all.ndata['feat'])
            pos_score = model(graph_all)
            neg_score = model(graph_neg_all)
            positive_weight = int(graph_neg_all.number_of_edges() / graph_all.number_of_edges())

            loss, f1 = compute_loss(pos_score, neg_score, positive_weight)

            # backward
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            if e % 10 == 0:

                print('In epoch {}, loss {}, f1 {}'.format(e, loss, f1))

    return model


def compute_probabilities_labels(results, ground_truth):

    probabilities = []
    labels = []


    for c11, c22, score in results:
        if ((fabricated_to_source_filename(c11[0]), c11[1]), (fabricated_to_source_filename(c22[0]), c22[1])) in\
                ground_truth or ((fabricated_to_source_filename(c22[0]), c22[1]), (fabricated_to_source_filename(c11[0]), c11[1])) in ground_truth\
                or (fabricated_to_source_filename(c11[0]) == fabricated_to_source_filename(c22[0]) and c11[1]==c22[1]):
            labels.append(1)
        else:
            labels.append(0)
        probabilities.append(score)

    return labels, probabilities

def compute_confusion_matrix(results, ground_truth, ground_truth_type, threshold=0.75):
    """
        Compute confusion matrix based on similarity scores and ground truth
    """
    count_tp = 0
    count_fp = 0
    count_tn = 0
    count_fn = 0

    with open('false-positives.txt', 'w') as file:
        if ground_truth_type == 'full':
            for c11, c22, score in results:
                if ((fabricated_to_source_filename(c11[0]), c11[1]), (fabricated_to_source_filename(c22[0]), c22[1])) in\
                        ground_truth or ((fabricated_to_source_filename(c22[0]), c22[1]), (fabricated_to_source_filename(c11[0]), c11[1])) in ground_truth\
                        or (fabricated_to_source_filename(c11[0]) == fabricated_to_source_filename(c22[0]) and c11[1]==c22[1]):
                    if score >= threshold:
                        count_tp += 1
                    else:
                        count_fn += 1
                else:
                    if score <= threshold:
                        count_tn += 1
                    else:
                        file.write('Dataset-1: {} -- Column-1: {} || Dataset-2:{} -- Column-1:{} = {} \n'.format(c11[0], c11[1], c22[0], c22[1], score))
                        count_fp += 1
        else:
            for c11, c22, score in results:
                if c11[1] == c22[1] or (c11[1], c22[1]) in ground_truth or (c22[1], c11[1]) in ground_truth:
                    if score >= threshold:
                        count_tp += 1
                    else:
                        count_fn += 1
                else:
                    if score <= threshold:
                        count_tn += 1
                    else:
                        file.write('Dataset-1: {} -- Column-1: {} || Dataset-2:{} -- Column-1:{} = {} \n'.format(c11[0], c11[1], c22[0], c22[1], score))
                        count_fp += 1

    return count_tp, count_fp, count_tn, count_fn


def predict_all_links(all_columns, all_cols_ids, embeddings, ground_truth, ground_truth_type, model, no_graphs, threshold=0.5):

    """
        Compute link predictions among all relatedness graphs and calculated effectiveness results.

        Output:
            Precision, recall and F1-score based on the ground_truth
    """

    cols = [i for i in range(no_graphs)]
    for c1, c2 in itertools.combinations(cols, 2):
        #print("Computing predictions between graphs: " + str(c1) + " - " + str(c2))

        no_threads = cpu_count() - 1
        similarities = run_multithread(all_columns[c1], all_columns[c2], embeddings[c1], embeddings[c2], all_cols_ids[c1],
                                       all_cols_ids[c2], model.predictor, no_threads)
        count_tp, count_fp, _, count_fn = compute_confusion_matrix(similarities, ground_truth, ground_truth_type, threshold)

    precision, recall, f1_score = metrics(count_tp, count_fp, count_fn)

    return precision, recall, f1_score

def compute_all_probs_labels_mlp(all_columns, all_cols_ids, embeddings, ground_truth, model, no_graphs):

    cols = [i for i in range(no_graphs)]
    all_probs = []
    all_labels = []
    for c1, c2 in itertools.combinations(cols, 2):
        print("Computing between graphs: " + str(c1) + " - " + str(c2))

        no_threads = cpu_count() - 1
        similarities = run_multithread(all_columns[c1], all_columns[c2], embeddings[c1], embeddings[c2], all_cols_ids[c1],
                                       all_cols_ids[c2], model, no_threads)
        labels, probabilities = compute_probabilities_labels(similarities, ground_truth)
        all_probs.extend(probabilities)
        all_labels.extend(labels)

    return np.array(all_labels), np.array(all_probs)

def compute_all_probs_labels(all_columns, all_cols_ids, embeddings, ground_truth, model, no_graphs):

    cols = [i for i in range(no_graphs)]
    all_probs = []
    all_labels = []
    all_similarities = []
    for c1, c2 in itertools.combinations(cols, 2):
        print("Computing between graphs: " + str(c1) + " - " + str(c2))

        no_threads = cpu_count() - 1
        similarities = run_multithread(all_columns[c1], all_columns[c2], embeddings[c1], embeddings[c2], all_cols_ids[c1],
                                       all_cols_ids[c2], model.predictor, no_threads)
        all_similarities.extend(similarities)
        labels, probabilities = compute_probabilities_labels(similarities, ground_truth)
        all_probs.extend(probabilities)
        all_labels.extend(labels)

    return np.array(all_labels), np.array(all_probs), all_similarities

def compute_starmie_probs_labels(all_columns, vectors, ground_truth, no_graphs):

    cols = [i for i in range(no_graphs)]

    all_probs = []
    all_labels = []
    for c1, c2 in itertools.combinations(cols, 2):
        
        print("Computing between graphs: " + str(c1) + " - " + str(c2))
        no_threads = cpu_count() - 1
        similarities = run_starmie_multithread(all_columns[c1], all_columns[c2], vectors, no_threads)
        labels, probabilities = compute_probabilities_labels(similarities, ground_truth)
        all_probs.extend(probabilities)
        all_labels.extend(labels)

    return np.array(all_labels), np.array(all_probs)
