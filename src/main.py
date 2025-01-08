import torch

from os.path import abspath, join, dirname
from torch.utils.data import DataLoader

from classes.streetsign_dataset import StreetsignDataset
from utils.data_reader import get_training_testing_data_2
from utils.tensors import load_image_tensors

tensors_path = abspath(join(dirname(__file__), "..", "data", "tensors"))

img_size = (25, 25)
if __name__ == '__main__':
    # load precalculated tensors
    tensors = load_image_tensors(tensors_path)

    # get training and testing split with 0.9 training
    training, testing = get_training_testing_data_2(tensors, 0.9)

    training_dataset = StreetsignDataset(training)
    testing_dataset = StreetsignDataset(testing)


    training_data_loader = DataLoader(training_dataset, batch_size=5, shuffle=True)
    testing_data_loader = DataLoader(testing_dataset, batch_size=5, shuffle=True)