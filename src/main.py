import csv
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import scipy.stats as stats
import multiprocessing
from src.utils.read_json import read_json_file_and_extract_vectors, get_training_testing_data_list, \
    concept_map_reversed2

# Filepaths
FILEPATH_TRAINING = 'C:/Users/Karsten/Documents/Programmierprojekte/ml1_Task3/data/training.json'
FILEPATH_CLASSIFICATION = 'C:/Users/Karsten/Documents/Programmierprojekte/ml1_Task3/data/classification_vectors.json'

def calculate_quality(n_clusters, training_data, training_labels, testing_data, predicted_labels):
    # Annotate clusters based on training data
    results = [[0] * 6 for _ in range(n_clusters)]
    for i, data in enumerate(training_data):
        results[training_labels[i]][data[0]] += 1

    cluster_annotations = [
        cluster.index(max(cluster)) if sum(cluster) > 0 else -1 for cluster in results
    ]

    small_clusters = 0
    clusters_per_sign = [0] * 6
    dominant_50_count, dominant_70_count = 0, 0

    for i, cluster in enumerate(results):
        cluster_sum = sum(cluster)

        if cluster_sum > 0:
            max_val = max(cluster)
            dominant_sign = cluster.index(max_val)
            clusters_per_sign[dominant_sign] += 1
            if cluster_sum <= 20:
                small_clusters += 1
            if max_val / cluster_sum >= 0.5:
                dominant_50_count += 1
                if max_val / cluster_sum >= 0.7:
                    dominant_70_count += 1

    # Evaluate testing data using cluster annotations
    correct, wrong = 0, 0
    for i, data in enumerate(testing_data):
        predicted_cluster = predicted_labels[i]
        actual_label = data[0]
        predicted_label = cluster_annotations[predicted_cluster]
        if predicted_label == actual_label:
            correct += 1
        else:
            wrong += 1

    quality = correct / (correct + wrong) if correct + wrong > 0 else 0

    return (
        quality,
        clusters_per_sign.count(0),
        small_clusters,
        dominant_50_count,
        dominant_70_count,
    )

def train_kmeans(n_clusters, training_data):
    x_train = np.array([t[1] for t in training_data])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(x_train)
    return kmeans.labels_, kmeans.cluster_centers_

def test_kmeans(n_clusters, centroids, training_data, training_labels, testing_data):
    x_test = np.array([t[1] for t in testing_data])
    predicted_labels = []
    total_sse = 0

    for point in x_test:
        distances = np.linalg.norm(centroids - point, axis=1)
        predicted_labels.append(np.argmin(distances))
        total_sse += np.min(distances) ** 2

    quality, no_clusters, small_clusters, dominant50, dominant70 = calculate_quality(
        n_clusters, training_data, training_labels, testing_data, predicted_labels
    )

    silhouette = silhouette_score(x_test, predicted_labels) if n_clusters > 1 else -1
    davies_bouldin = davies_bouldin_score(x_test, predicted_labels) if n_clusters > 1 else -1
    calinski_harabasz = calinski_harabasz_score(x_test, predicted_labels) if n_clusters > 1 else -1

    return total_sse, quality, no_clusters, small_clusters, silhouette, davies_bouldin, calinski_harabasz, dominant50, dominant70

def save_results_to_csv(filename, results):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(
            [
                str(item).replace('.', ',') if isinstance(item, float) else item
                for item in results
            ]
        )

def main(filename, runs = 100, n_clusters_start = 2, n_clusters_end = 100, data_source = '', new_file = True):

    # read data source
    if data_source == '':
        data = read_json_file_and_extract_vectors(FILEPATH_TRAINING)
        data.extend(read_json_file_and_extract_vectors(FILEPATH_CLASSIFICATION))
    else:
        with open('C:\\Users\\Karsten\\Documents\\Programmierprojekte\\ml1_Task3\\data\\' + data_source + '.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)
            concepts_list = data_dict['concepts'].tolist()
            for i in range(len(concepts_list)):
                concepts_list[i] = concept_map_reversed2.get(concepts_list[i], -1)

            vectors_list = data_dict['vectors'].tolist()
            data = list(zip(concepts_list, vectors_list))

    # prepare output file
    output_file = filename + '.csv'

    if new_file:
        if os.path.exists(output_file):
            os.remove(output_file)

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(
                ['N_Cluster', 'Quality', 'c_lower', 'c_upper', 'NoClusters', 'SmallClusters', '>=50% Clusters', '>=70% Clusters', 'SSE', 'Silhouette', 'DaviesBouldin', 'CalinskiHarabasz']
            )

    for n_clusters in range(n_clusters_start, n_clusters_end + 1):
        process_n_clusters(n_clusters, runs, data)

    #with multiprocessing.Pool() as pool:
    #    pool.starmap(process_n_clusters, [(n, runs, data) for n in range(n_clusters_start, n_clusters_end + 1)])


def process_n_clusters(n_clusters, runs, data):
    print(str(n_clusters) + ' start')

    # collect data
    aggregate_results = np.zeros(9)
    quality_values = []

    for _ in range(runs):
        training_data, testing_data = get_training_testing_data_list(data, 0.8)

        training_labels, centroids = train_kmeans(n_clusters, training_data)
        sse, quality, no_clusters, small_clusters, silhouette, davies_bouldin, calinski_harabasz, dominant50, dominant70 = test_kmeans(
            n_clusters, centroids, training_data, training_labels, testing_data
        )

        quality_values.append(quality)

        aggregate_results += [
            sse,
            quality,
            no_clusters / 6,
            small_clusters / n_clusters,
            dominant50 / n_clusters,
            dominant70 / n_clusters,
            silhouette,
            davies_bouldin,
            calinski_harabasz,
        ]

    mean_quality = np.mean(quality_values)
    std_error = stats.sem(quality_values)
    ci = stats.t.interval(0.95, len(quality_values) - 1, loc=mean_quality, scale=std_error)

    avg_results = aggregate_results / runs

    print(n_clusters)

    save_results_to_csv('output_karsten.csv', [n_clusters, mean_quality, ci[0], ci[1], *avg_results[1:]])


if __name__ == '__main__':
    datasources = ['', 'bene', 'denys', 'jonathan', 'karsten']

    for datasource in datasources:
        output_file = 'output_' + datasource
        main(output_file, data_source=datasource, n_clusters_start=44)