from collections import OrderedDict
import pandas as pd
import os
from csv import DictReader
import json
import fasttext
import numpy as np
import torch
import random
import dgl
import torch.nn.functional as F
from multiprocessing import Pool
import itertools
import torch.nn as nn
import ast
import string
import math
from scipy.stats import skew, kurtosis
import nltk
import re
import pickle

# ------------- FUNCTIONS IMPORTED FROM SHERLOCK ---------------


def extract_bag_of_characters_features(data):
    characters_to_check = (
            ['[' + c + ']' for c in string.printable if c not in ('\n', '\\', '\v', '\r', '\t', '^')]
            + ['[\\\\]', '[\^]']
    )

    f = OrderedDict()

    data_no_null = data.dropna()
    all_value_features = OrderedDict()

    for c in characters_to_check:
        all_value_features['n_{}'.format(c)] = data_no_null.str.count(c)

    for value_feature_name, value_features in all_value_features.items():
        f['{}-agg-any'.format(value_feature_name)] = any(value_features)
        f['{}-agg-all'.format(value_feature_name)] = all(value_features)
        f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
        f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
        f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
        f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
        f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
        f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
        f['{}-agg-kurtosis'.format(value_feature_name)] = kurtosis(value_features)
        f['{}-agg-skewness'.format(value_feature_name)] = skew(value_features)

    return f


def extract_bag_of_words_features(data, n_val):
    f = OrderedDict()
    data = data.dropna()

    # n_val = data.size

    if not n_val: return

    # Entropy of column
    freq_dist = nltk.FreqDist(data)
    probs = [freq_dist.freq(l) for l in freq_dist]
    f['col_entropy'] = -sum(p * math.log(p, 2) for p in probs)

    # Fraction of cells with unique content
    num_unique = data.nunique()
    f['frac_unique'] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    num_cells = np.sum(data.str.contains('[0-9]', regex=True))
    text_cells = np.sum(data.str.contains('[a-z]|[A-Z]', regex=True))
    f['frac_numcells'] = num_cells / n_val
    f['frac_textcells'] = text_cells / n_val

    # Average + std number of numeric tokens in cells
    num_reg = '[0-9]'
    f['avg_num_cells'] = np.mean(data.str.count(num_reg))
    f['std_num_cells'] = np.std(data.str.count(num_reg))

    # Average + std number of textual tokens in cells
    text_reg = '[a-z]|[A-Z]'
    f['avg_text_cells'] = np.mean(data.str.count(text_reg))
    f['std_text_cells'] = np.std(data.str.count(text_reg))

    # Average + std number of special characters in each cell
    spec_reg = '[[!@#$%^&*(),.?":{}|<>]]'
    f['avg_spec_cells'] = np.mean(data.str.count(spec_reg))
    f['std_spec_cells'] = np.std(data.str.count(spec_reg))

    # Average number of words in each cell
    space_reg = '[" "]'
    f['avg_word_cells'] = np.mean(data.str.count(space_reg) + 1)
    f['std_word_cells'] = np.std(data.str.count(space_reg) + 1)

    all_value_features = OrderedDict()

    data_no_null = data.dropna()

    f['n_values'] = n_val

    all_value_features['length'] = data_no_null.apply(len)

    for value_feature_name, value_features in all_value_features.items():
        f['{}-agg-any'.format(value_feature_name)] = any(value_features)
        f['{}-agg-all'.format(value_feature_name)] = all(value_features)
        f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
        f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
        f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
        f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
        f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
        f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
        f['{}-agg-kurtosis'.format(value_feature_name)] = kurtosis(value_features)
        f['{}-agg-skewness'.format(value_feature_name)] = skew(value_features)

    n_none = data.size - data_no_null.size - len([e for e in data if e == ''])
    f['none-agg-has'] = n_none > 0
    f['none-agg-percent'] = n_none / len(data)
    f['none-agg-num'] = n_none
    f['none-agg-all'] = (n_none == len(data))

    return f


# -----------------------------------------------------

def create_configuration(cols_to_ids, min_tables_per_silo, max_tables_per_silo, number_of_silos, con_type):
    config = dict()

    cols = list(cols_to_ids.keys())

    base_derived = dict()

    for c in cols:
        match = re.match(r"([a-z0-9_]+)_([0-9]+)", c[0], re.I)   
        if match[1] in base_derived:
            base_derived[match[1]].append(c[0])
        else:
            base_derived[match[1]] = [c[0]]

    for c,t in base_derived.items():    
        t = list(set(t))
        base_derived[c] = t

    if con_type == 'incremental':
        random_bases = list(base_derived.keys())
        random.shuffle(random_bases)


        for i in range(number_of_silos):
            tables_per_silo  = max_tables_per_silo
            config[i] = dict()

            for j in range(0, i+1):
                previous_right  = 0
                if j == i:
                    config[i][random_bases[j]] = [0, tables_per_silo]
                else:
                    if i > 0:
                        previous_right = config[i-1][random_bases[j]][1]
                    config[i][random_bases[j]] = [previous_right, previous_right + tables_per_silo]
    elif con_type == 'case1':
        i = 0
        for base, synthetic in base_derived.items():
            config[i] = dict()
            random.shuffle(synthetic)

            no_datasets = random.randint(min_tables_per_silo, max_tables_per_silo)

            config[i][base] = [0, no_datasets]
            i += 1
    elif con_type == 'case2':
        random_bases = list(base_derived.keys())
        random.shuffle(random_bases)
        last = 0
        increment = len(random_bases) // number_of_silos
        for i in range(number_of_silos):
            config[i] = dict()
            for j in range(last, last + increment):

                random.shuffle(base_derived[random_bases[j]])
                config[i][random_bases[j]] = [0, random.randint(min_tables_per_silo, max_tables_per_silo)]
            last = last + increment
        if last <= len(random_bases):
            for j in range(last, len(random_bases)):

                random.shuffle(base_derived[random_bases[j]])

                config[i][random_bases[j]] = [0, random.randint(min_tables_per_silo, max_tables_per_silo)]
    elif con_type == 'random':
        bases = list(base_derived.keys())
        num_bases = len(base_derived)
        last_accessed = {i: 0 for i in range(num_bases)}

        for i in range(number_of_silos):
            config[i] = dict()

            base_indexes = random.sample(range(num_bases), random.randint(1, num_bases))

            for index in base_indexes:

                tables = random.choice(range(min_tables_per_silo, max_tables_per_silo+1))

                config[i][bases[index]] = [last_accessed[index], last_accessed[index] + tables]
                last_accessed[index] += tables
    else:
        bases = list(base_derived.keys())
        random.shuffle(bases)
        num_bases = len(base_derived)
        last_accessed = {i: 0 for i in range(num_bases)}

        for i in range(number_of_silos):
            config[i]  = dict()

            base_indexes = [i, (i+1)%number_of_silos, (i+2)%number_of_silos]

            for index in base_indexes:

                tables = random.choice(range(min_tables_per_silo, max_tables_per_silo+1))

                config[i][bases[index]] = [last_accessed[index], last_accessed[index] + tables]
                last_accessed[index] += tables


    refined_config = dict()

    for silo, base_range in config.items():
        refined_config[silo] = set()
        for base, range_index in base_range.items():
            for i in range(range_index[0], range_index[1]):
                refined_config[silo].add(base_derived[base][i])


    return refined_config

def csvs_to_dataframes(dir_path: str):
    dataframe_dict = dict()
    
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(dir_path, filename))
            
            columns = df.columns.tolist()
            
            for col in columns:
                df.rename(columns={col: col.rstrip()}, inplace=True)
            
            dataframe_dict[filename] = df
    return dataframe_dict

def get_features(data: pd.DataFrame) -> pd.DataFrame:
    """
        Code for profiling tabular datasets and based on the profiler of Sherlock.

        Input:
            data: A pandas DataFrame with each row a list of string values
        Output:
            a dataframe where each row represents a column and columns represent the features
            computed for the corresponding column.
    """

    # Transform data so that each column becomes a row with its corresponding values as a list

    data = data.T
    list_values = data.values.tolist()
    data = pd.DataFrame(data={'values': list_values})

    data_columns = data['values']

    features_list = []

    for column in data_columns:
        column = pd.Series(column).astype(str)

        f = OrderedDict(list(extract_bag_of_characters_features(column).items()) + list(
            extract_bag_of_words_features(column, len(column)).items()))

        features_list.append(f)

    return pd.DataFrame(features_list).reset_index(drop=True) * 1


def create_profiles_tensor(col_profiles: dict(), col_ids: dict()):
    """

    :param col_profiles: Correspondences between columns and their profiles
    :param col_ids: Correspondences between columns and ids
    :return: Correspondence between col ids and profiles in the form of a tensor
    """

    profiles_per_column = [[]] * len(col_ids)


    for col, id in col_ids.items():
        profiles_per_column[id] = col_profiles[col]

    return torch.tensor(profiles_per_column, dtype=torch.float)

def get_coma_results(results_file: str, all_cols_ids, all_pairs):
    """
        Returns a list containing the matching results of the matching method to which the
        input .json file belongs.
    """

    def _parse_tuple(string):  # simple function to parse tuples
        try:
            s = ast.literal_eval(str(string))
            if type(s) == tuple:
                return s
            return
        except:
            return

    with open(results_file) as f:
        results_dict = json.load(f)

    results = []

    for dict_score in results_dict:
        for k, v in dict_score.items():
            kt = _parse_tuple(k)
            kt1, kt2 = kt
            results.append(((kt1[0][:-4], kt1[1]), (kt2[0][:-4], kt2[1]), v))

    # Normalize similarity scores
    results.sort(key=lambda tup: tup[2])
    amin, amax = results[0][2], results[-1][2]
    for i, val in enumerate(results):
        results[i] = (val[0], val[1], max(0, (val[2] - amin) / (amax - amin)))

    # keep results only for the columns included in the silo configuration and belong to different silos
    col_to_silo = dict()

    for i, cols_to_ids in all_cols_ids.items():
        for col in cols_to_ids:
            col_to_silo[col] = i

    filtered_results = []
    all_pairs_coma = set()

    for c1, c2, score in results:
        if c1 in col_to_silo and c2 in col_to_silo:
            if col_to_silo[c1] != col_to_silo[c2]:
                filtered_results.append((c1, c2, score))
                all_pairs_coma.add((c1, c2))

    # make results complete for pairs that coma didn't output results for

    for pair in all_pairs:
        if pair not in all_pairs_coma and (pair[1], pair[0]) not in all_pairs_coma:
            filtered_results.append((pair[0], pair[1], 0))

    return filtered_results

def get_starmie_vectors(dataset_path, embeddings_path):
   
    dataframes = csvs_to_dataframes(dataset_path)
    
    with open(embeddings_path, 'rb') as file:
            v = pickle.load(file)

    file_cols_vectors = dict()
    for file, vectors in v:
        file_cols_vectors[file] = dict()
        for i in range(len(vectors)):
            file_cols_vectors[file][dataframes[file].columns[i]] = vectors[i]    

    return file_cols_vectors


def get_fasttext_embeddings(values, model_file):
    """
        Compute pre-trained embeddings for a list of values.
    """
    model = fasttext.load_model(model_file)

    f = OrderedDict()
    embeddings = []

    values = values.dropna()

    for v in values:

        v = str(v).lower()

        vl = v.split(' ')

        if len(vl) == 1:
            embeddings.append(model.get_word_vector(v))
        else:
            embeddings_to_all_words = []

            for w in vl:
                embeddings_to_all_words.append(model.get_word_vector(w))

            mean_of_word_embeddings = np.nanmean(embeddings_to_all_words, axis=0)
            embeddings.append(mean_of_word_embeddings)

    mean_embeddings = np.nanmean(embeddings, axis=0)

    for i, e in enumerate(mean_embeddings): f['word_embedding_avg_{}'.format(i)] = e

    return f


def get_embeddings(data: pd.DataFrame, model_file) -> pd.DataFrame:
    """
        Compute pre-trained FastText embeddings for each column of a tabular dataset,
        represented as a pandas dataframe.
    """
    data = data.T
    list_values = data.values.tolist()
    data = pd.DataFrame(data={'values': list_values})

    data_columns = data['values']

    embeddings_list = []

    for column in data_columns:
        column = pd.Series(column).astype(str)

        f = OrderedDict(list(get_fasttext_embeddings(column, model_file).items()))

        embeddings_list.append(f)

    return pd.DataFrame(embeddings_list).reset_index(drop=True) * 1


def generate_ids_paths(dir_path: str) -> (dict(), dict()):
    """
        Connects each column in the dataset to an id, and each file to each full_path

        Input:
            dir_path: The directory path under which we search for .csv datasets
        Output:
            columns_to_ids: A matching between columns and ids
            files_to_paths: A matching between filenames and full paths
    """

    columns_to_ids = dict()
    files_to_paths = dict()
    column_id = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".csv"):
                files_to_paths[file[:-4]] = os.path.join(root, file)
                with open(os.path.join(root, file), 'r') as read_obj:
                    csv_dict_reader = DictReader(read_obj)
                    column_names = csv_dict_reader.fieldnames

                    for c in column_names:
                        columns_to_ids[(file[:-4], c)] = column_id
                        column_id += 1

    return columns_to_ids, files_to_paths

def columns_to_profiles(files_to_paths: dict()) -> dict():
    """
        Input:
            files_to_paths: Correspondences between files and their full paths
        Output:
            col_features: dictionary with column - profile correspondence
    """

    col_profiles = dict()
    count = 1
    cols_count = len(files_to_paths)
    for file, filepath in files_to_paths.items():

        print('File ' + str(count) + "/" + str(cols_count))

        count += 1

        file_pd = pd.read_csv(filepath)

        cols = file_pd.columns.tolist()

        col_profiles_pd = get_features(file_pd)

        profiles_list = col_profiles_pd.values.tolist()

        for i in range(len(cols)):
            col_profiles[(file, cols[i])] = profiles_list[i]

    return col_profiles

def metrics(count_tp, count_fp, count_fn):
    precision = (count_tp * 1.0) / (count_tp + count_fp)
    recall = (count_tp * 1.0) / (count_tp + count_fn)
    f1_score = 2 * precision * recall / (precision + recall)
    #print('Precision: ' + str(precision))
    #print('Recall: ' + str(recall))
    #print('F-score: ' + str(f_score))
    return precision, recall, f1_score


def fabricated_to_source_filename(fabricated_name: str) -> str:
    """
    Simple function that returns the name of the source on which the fabricated dataset is based. It uses info on the
    naming conventions we use for the fabricated files, i.e. sourcename_[clean|noisy]_id.csv
    :param fabricated_name: The filename of the fabricated dataset
    :return: The name of the source -> sourcename
    """

    source_name = re.split('_clean_|_noisy_', fabricated_name)[0]

    return source_name

############ Functions for multiprocessing of GNN results############

def prediction_score(h1, h2, c11, c22, cid1, cid2, pred):
    first = h1[cid1[c11]]

    second = h2[cid2[c22]]
    hh = torch.cat([first, second])
    score = torch.sigmoid(pred.W2(F.relu(pred.W1(hh)))).detach().item()
    return score


def get_process_pairs(columns1, columns2, h1, h2, cid1, cid2, pred):
    for c11, c22 in itertools.product(columns1, columns2):
        yield c11, c22, h1, h2, cid1, cid2, pred


def process_score(input_tuple):
    c11, c22, h1, h2, cid1, cid2, pred = input_tuple
    score = prediction_score(h1, h2, c11, c22, cid1, cid2, pred)
    return c11, c22, score


def run_multithread(columns1, columns2, h1, h2, cid1, cid2, pred, no_threads):
    with Pool(no_threads) as process_pool:
        similarities = process_pool.map(process_score, get_process_pairs(columns1, columns2, h1, h2, cid1, cid2, pred))

    return similarities


############ Functions for multiprocessing of baseline results############


def get_baseline_pairs(columns1, columns2, h1, h2, cid1, cid2):
    for c11, c22 in itertools.product(columns1, columns2):
        yield c11, c22, h1, h2, cid1, cid2


def baseline_score(h1, h2, c11, c22, cid1, cid2):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    first = h1[cid1[c11]]

    second = h2[cid2[c22]]

    return cos(first, second).detach().item()


def baseline_process_score(input_tuple):
    c11, c22, h1, h2, cid1, cid2 = input_tuple
    score = baseline_score(h1, h2, c11, c22, cid1, cid2)
    return c11, c22, score


def run_baseline_multithread(columns1, columns2, h1, h2, cid1, cid2, no_threads):
    with Pool(no_threads) as process_pool:
        similarities = process_pool.map(baseline_process_score,
                                        get_baseline_pairs(columns1, columns2, h1, h2, cid1, cid2))

    return similarities


########## Functions for multiprocessing of starmie results ############

def get_starmie_pairs(columns1, columns2, vec):
    for c11, c22 in itertools.product(columns1, columns2):
        yield c11, c22, vec

def starmie_score(c11, c22, vec):
   
  
    file1 = c11[0] + '.csv'  
    file2 = c22[0] + '.csv'


    col1 = c11[1]
    col2 = c22[1]

    vec1 = vec[file1][col1]
    vec2 = vec[file2][col2]

    return np.dot(vec1, vec2)/(norm(vec1)*norm(vec2))

def starmie_process_score(input_tuple):
    c11, c22, vec = input_tuple
    score = starmie_score(c11, c22, vec)
    return c11, c22, score

def run_starmie_multithread(columns1, columns2, vec, no_threads):
    with Pool(no_threads) as process_pool:
        similarities = process_pool.map(starmie_process_score, get_starmie_pairs(columns1, columns2, vec))

    return similarities