from sklearn.cluster import KMeans
import torch

from os.path import abspath, join, dirname
from utils.data_reader import get_training_testing_data_2
from utils.tensors import load_image_tensors

tensors_path = abspath(join(dirname(__file__), "..", "data", "tensors"))

number_of_clusters = 6

if __name__ == '__main__':
    # load precalculated tensors
    tensors = load_image_tensors(tensors_path)

    # get training and testing split with 0.9 training
    training, testing = get_training_testing_data_2(tensors, 0.9)

    kmeans = KMeans(n_clusters=number_of_clusters)