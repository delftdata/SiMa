import torch
import dgl
import random
import torch.nn.functional as F
from tools import fabricated_to_source_filename


def create_relatedness_graphs(base_to_derived, cols_to_ids, profiles_list, ground_truth, silo_config):


    """
        Function to construct relatedness graphs.

        Input:
            base_to_derived: dictionary which stores source_table - fabricated datasets correspondences
            cols_to_ids: dictionary with columns to ids correspondences
            profiles_list: stores for each node id the corresponding profile

    """
    no_silos = len(silo_config)

    datasets_per_silo = {i: [] for i in range(no_silos)}

    for silo_num, datasets_per_base in silo_config.items():

        for base, index_range in datasets_per_base.items():
            datasets_per_silo[silo_num].extend(base_to_derived[base][index_range[0]:index_range[1]])

    columns = {i: [] for i in range(no_silos)}

    for k, _ in cols_to_ids.items():

        for i, s in datasets_per_silo.items():
            if k[0] in s:
                columns[i].append(k)
                break

    all_cols_ids = dict()
    all_ids_cols = dict()

    for i, col in columns.items():
        count = 0
        d = dict()
        for c in col:
            d[c] = count
            count += 1
        all_cols_ids[i] = d
        invd = {v: k for k, v in d.items()}
        all_ids_cols[i] = invd

    profiles = {i: [[]] * len(columns[i]) for i in range(no_silos)}

    for i in range(no_silos):
        for c in columns[i]:
            profiles[i][all_cols_ids[i][c]] = profiles_list[cols_to_ids[c]]

    edges = {i: [] for i in range(no_silos)}


    for i, cols in columns.items():
        for j in range(len(cols)):
            matched = False
            for k in range(j + 1, len(cols)):
                if ((fabricated_to_source_filename(cols[j][0]), cols[j][1]), (fabricated_to_source_filename(cols[k][0]), cols[k][1])) in ground_truth or (
                        (fabricated_to_source_filename(cols[k][0]), cols[k][1]), (fabricated_to_source_filename(cols[j][0]), cols[j][1])) in ground_truth\
                        or (fabricated_to_source_filename(cols[j][0]) ==  fabricated_to_source_filename(cols[k][0]) and
                            cols[j][1] == cols[k][1]):
                    matched = True
                    edge1 = (all_cols_ids[i][cols[j]], all_cols_ids[i][cols[k]])
                    edge2 = (edge1[1], edge1[0])
                    edges[i].append(edge1)
                    edges[i].append(edge2)
            if not matched:
                edges[i].append((all_cols_ids[i][cols[j]], all_cols_ids[i][cols[j]]))


    graphs = dict()

    for i in range(no_silos):
        edges_tensor = torch.tensor(edges[i], dtype=torch.long).t().contiguous()
        profiles_tensor = torch.tensor(profiles[i], dtype=torch.float)
        graph = dgl.graph((edges_tensor[0], edges_tensor[1]))
        graph.ndata['feat'] = F.normalize(profiles_tensor, 2, 0)  # normalize input profiles
        graphs[i] = graph

    return graphs, columns, all_cols_ids, all_ids_cols
