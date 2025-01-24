import numpy as np
from sklearn.cluster import KMeans
import torch

from os.path import abspath, join, dirname

from src.utils.read_json import read_json_file_and_extract_vectors, get_training_testing_data_list

tensors_path = abspath(join(dirname(__file__), "..", "data", "data"))

number_of_clusters = 6


def calculate_quality(training_data, label_list):
    result = [[0, 0, 0, 0, 0, 0] for _ in range(number_of_clusters)]
    for i in range(len(training_data)):
        result[label_list[i]][training_data[i][0]] += 1

    quality_list = []
    count_empty_clusters = 0
    correct_classified = 0
    wrong_classified = 0
    for i in range(len(result)):
        if sum(result[i]) != 0:
            quality_list.append((max(result[i]) / sum(result[i])))
            correct_classified += max(result[i])
            wrong_classified += sum(result[i]) - max(result[i])
        else:
            count_empty_clusters += 1
        #print(result[i])

    quality_list.sort()
    print(quality_list)
    print(f'empty clusters: {count_empty_clusters}')
    print(f'correct classified: {correct_classified}, wrong classified: {wrong_classified}')
    print(f'model: {correct_classified/(wrong_classified+correct_classified)}')


def train(training_data):
    x_train = np.array([np.array(t[1]) for t in training_data])
    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(x_train)
    return kmeans.labels_, kmeans.cluster_centers_

def test(centroid_list, testing_data):
    x_test = np.array([np.array(t[1]) for t in testing_data])

    concepts = [t[0] for t in testing_data]

    predicted_concepts = []
    total_sse = 0
    for data_point in x_test:
        distances = [np.linalg.norm(data_point - centroid) for centroid in centroid_list]
        predicted_concepts.append(np.argmin(distances))
        total_sse += min(distances) ** 2  # Summe der quadratischen Fehler berechnen

    calculate_quality(testing_data, predicted_concepts)

    # Berechnung von SSD (Sum of Squared Distances between Centroids) und Gesamtqualität
    cluster_distances = [
        np.linalg.norm(centroid1 - centroid2) ** 2
        for i, centroid1 in enumerate(centroid_list)
        for j, centroid2 in enumerate(centroid_list)
        if i < j
    ]
    total_ssd = sum(cluster_distances)

    print(f"Total SSE (Sum of Squared Errors): {total_sse}")
    print(f"Total SSD (Sum of Squared Distances between Centroids): {total_ssd}")


if __name__ == '__main__':
    data = [] #TODO

    training, testing = get_training_testing_data_list(data, 0.8)

    print(f"Training List: {len(training)}")
    print(f"Testing List: {len(testing)}")

    labels, centroids = train(training)

    # print("Clusterzentren:")
    # print(centroids)
    # print("Labels:")
    # print(labels)

    # calculate_quality(training, labels)

    test(centroids, testing)