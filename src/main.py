import csv
import os
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.utils.read_json import read_json_file_and_extract_vectors, get_training_testing_data_list

# Filepaths
FILEPATH_TRAINING = ''
FILEPATH_CLASSIFICATION = ''

def calculate_quality(n_clusters, training_data, training_labels, testing_data, predicted_labels):
    # Annotate clusters based on training data
    results = [[0] * 6 for _ in range(n_clusters)]
    for i, data in enumerate(training_data):
        results[training_labels[i]][data[0]] += 1

    cluster_annotations = [
        cluster.index(max(cluster)) if sum(cluster) > 0 else -1 for cluster in results
    ]

    correct, wrong, empty_clusters, small_clusters = 0, 0, 0, 0
    clusters_per_sign = [0] * 6

    for i in range(len(results)):
        if sum(results[i]) > 0:
            clusters_per_sign[results[i].index(max(results[i]))] += 1
            if sum(results[i]) <= 20:
                small_clusters += 1
        else:
            empty_clusters += 1

    # Evaluate testing data using cluster annotations
    test_correct, test_wrong = 0, 0
    for i, data in enumerate(testing_data):
        predicted_cluster = predicted_labels[i]
        actual_label = data[0]
        predicted_label = cluster_annotations[predicted_cluster]
        if predicted_label == actual_label:
            test_correct += 1
        else:
            test_wrong += 1

    quality = test_correct / (test_correct + test_wrong) if test_correct + test_wrong > 0 else 0

    return (
        quality,
        clusters_per_sign.count(0),
        empty_clusters,
        small_clusters,
    )

def train_kmeans(n_clusters, training_data):
    x_train = np.array([t[1] for t in training_data])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
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

    total_ssd = sum(
        np.linalg.norm(c1 - c2) ** 2
        for i, c1 in enumerate(centroids)
        for j, c2 in enumerate(centroids)
        if i < j
    )

    quality, no_clusters, empty_clusters, small_clusters = calculate_quality(
        n_clusters, training_data, training_labels, testing_data, predicted_labels
    )

    silhouette = silhouette_score(x_test, predicted_labels) if n_clusters > 1 else -1
    davies_bouldin = davies_bouldin_score(x_test, predicted_labels) if n_clusters > 1 else -1

    # Habarés-Index: Ratio of intra-cluster variance to inter-cluster variance
    inter_cluster_variance = total_ssd / (n_clusters * (n_clusters - 1) / 2) if n_clusters > 1 else 0
    intra_cluster_variance = total_sse / len(x_test) if len(x_test) > 0 else 0
    habares_index = intra_cluster_variance / inter_cluster_variance if inter_cluster_variance > 0 else -1

    return total_sse, total_ssd, quality, no_clusters, empty_clusters, small_clusters, silhouette, davies_bouldin, habares_index

def save_results_to_csv(filename, results):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            [
                str(item).replace(".", ",") if isinstance(item, float) else item
                for item in results
            ]
        )

def main():
    runs = 100
    n_clusters_start, n_clusters_end = 6, 100
    output_file = "output.csv"

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            ["SSE", "SSD", "Quality", "Time", "NoClusters", "EmptyClusters", "SmallClusters", "Silhouette", "DaviesBouldin", "Habares"]
        )

    data = read_json_file_and_extract_vectors(FILEPATH_TRAINING)
    data.extend(read_json_file_and_extract_vectors(FILEPATH_CLASSIFICATION))

    for n_clusters in range(n_clusters_start, n_clusters_end + 1):
        aggregate_results = np.zeros(10)
        for _ in range(runs):
            training_data, testing_data = get_training_testing_data_list(data, 0.8)
            training_labels, centroids = train_kmeans(n_clusters, training_data)
            start_time = time.time()

            sse, ssd, quality, no_clusters, empty_clusters, small_clusters, silhouette, davies_bouldin, habares_index = test_kmeans(
                n_clusters, centroids, training_data, training_labels, testing_data
            )

            elapsed_time = time.time() - start_time
            aggregate_results += [
                sse / 5000,
                ssd / 5000,
                quality,
                elapsed_time,
                no_clusters,
                empty_clusters,
                small_clusters,
                silhouette,
                davies_bouldin,
                habares_index,
            ]

        avg_results = aggregate_results / runs
        save_results_to_csv(output_file, avg_results)


if __name__ == "__main__":
    main()