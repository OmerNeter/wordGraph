import codecs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import random


def take_random(n, iterable, seed):
    random.seed(seed)
    return random.sample(iterable, n)


def txt_to_dict(root, start_position, vector_size):
    dict = {}
    with codecs.open(root, 'r', encoding="utf-8", errors='replace') as file:
        tokens_lst = list(map(lambda y: y.split(), file.readlines()))
        tokens = tuple([word for row_lst in tokens_lst for word in row_lst])
        for x in range(start_position, len(tokens), vector_size+1):
            dict[tokens[x]] = tuple(tokens[x+1: x+vector_size+1])
    return dict


def make_matrix(df):
    matrix = np.asarray(list(df['coordinates'].values), dtype=np.float)
    return matrix


def make_average_vector(df):
    matrix = make_matrix(df)
    return matrix.mean(0)


def distances(matrix):
    lst = []
    for i in range(0, matrix.shape[0]):
        B = np.asmatrix(list(matrix[i:, :]), dtype=np.float)
        w = np.asmatrix(list(matrix[i, :]), dtype=np.float).transpose()
        b = B * w
        lst.append(pd.DataFrame(b[1:,:], columns=['distances']))
    distances_df = pd.concat(lst)
    return distances_df

def get_average_distances(matrix):
    dist_tbl_matrix = distances(matrix)
    expectation = dist_tbl_matrix['distances'].mean()
    variance = dist_tbl_matrix['distances'].var()
    return expectation, variance

def chart(matrix, bins):
    #dist_tbl_matrix = distances(matrix)

    #dist = dist_tbl_matrix.apply(lambda x: np.exp(np.sqrt(2 - 2 * x)/2))
    #dist = dist_tbl_matrix.apply(lambda x: np.arccos(x)/2)
    dist_try = pd.DataFrame(np.random.normal(0.6, np.sqrt(0.006), 49995000), columns=["distances"])

    out_matrix = pd.cut(list(matrix['distances']), bins=bins, include_lowest=True)
    out_matrix_try = pd.cut(list(dist_try['distances']), bins=bins, include_lowest=True)

    out_bins_count_matrix = pd.value_counts(out_matrix, sort=False, normalize=True).reset_index()
    out_bins_count_matrix_try = pd.value_counts(out_matrix_try, sort=False, normalize=True).reset_index()

    out_bins_count_matrix.columns = ['distance_intervals', 'f_x']
    out_bins_count_matrix_try.columns = ['distance_intervals', 'f_x']

    out_bins_count_matrix['F_x'] = out_bins_count_matrix[out_bins_count_matrix.columns[1]].rolling(min_periods=1, window=out_bins_count_matrix.shape[0]).sum()
    out_bins_count_matrix_try['F_x'] = out_bins_count_matrix_try[out_bins_count_matrix.columns[1]].rolling(min_periods=1, window=out_bins_count_matrix_try.shape[0]).sum()

    out_bins_count_matrix.plot(kind='line', x='distance_intervals', y=['f_x'])
    plt.xticks(rotation=90)
    plt.show()
    out_bins_count_matrix.plot(kind='line', x='distance_intervals', y=['F_x'])
    plt.xticks(rotation=90)
    plt.show()
    # out_bins_count_matrix_try.plot(kind='line', x='distance_intervals', y=['f_x','F_x'])
    # plt.xticks(rotation=90)
    # plt.show()
    # plt.scatter(out_bins_count_matrix["f_x"], out_bins_count_matrix_try["f_x"])
    # plt.show()
    # print('d')


def mat_distances(matrix):
    lst = []
    for i in range(0, matrix.shape[0]):
        v = np.asmatrix(list(matrix[i, :]), dtype=np.float)
        M = np.linalg.norm(matrix[i:,:] - v, 2, axis=1)
        dist = pd.DataFrame(M, columns=['distances'])
        lst.append(dist)
    return pd.concat(lst, sort=False)

root = "C:\\Users\\User\\PycharmProjects\\NLP\\" + "model.txt"
all_items = txt_to_dict(root, 2, 100)
random_items = take_random(10000, all_items.items(), 7)

random_sampled_data = pd.DataFrame(random_items, columns=["words", "coordinates"])
all_data_tbl = pd.DataFrame(all_items.items(), columns=["words", "coordinates"])

average_vector = make_average_vector(all_data_tbl)
random_sampled_matrix = make_matrix(random_sampled_data)

matrix__minus_average = random_sampled_matrix - average_vector

matrix_minus_average_normalized = preprocessing.normalize(matrix__minus_average, norm='l2')
random_sampled_matrix_normalized = preprocessing.normalize(random_sampled_matrix, norm='l2')

#bins = list(np.arange(0, 12, 0.001))

ex_minus, var_minus = get_average_distances(random_sampled_matrix)
ex, var = get_average_distances(matrix__minus_average)

distances = mat_distances(random_sampled_matrix)
bins = list(np.arange(0, int(np.round(distances.max())), 0.001))

chart(distances, bins)
chart(matrix_minus_average_normalized, bins)

print('done')
# covariance matrix.
# given N100(mu,cov) what is E(d^2) and E(cos).

# maybe we need to substract the average vector of all the vectors.
# average vecotr.
# remove ones from diagonal of the matrix.
