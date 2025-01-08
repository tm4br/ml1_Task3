import os
import torch
import random

from classes.concept import Concept

# takes a dictionary with the concept as key and a list of tensors as value
# returns training_data and testing data, each as a list of tuples, containing concept and tensor
def get_training_testing_data_2(tensors_dict: dict[Concept, list[torch.Tensor]], training_percentage):
	training_data = []
	testing_data = []

	for concept in tensors_dict:
		tensors = tensors_dict[concept]
		random.shuffle(tensors)

		concept_tensor = []
		for tensor in tensors:
			concept_tensor.append((concept, tensor))

		pivot = int(len(concept_tensor)*training_percentage)

		training_data += concept_tensor[:pivot]
		testing_data += concept_tensor[pivot:]

	return training_data, testing_data


# reads all images from img_dir and splits them up into 
# training and testing data
def get_training_testing_data(img_dir, training_percentage):
	training_data = []
	testing_data = []

	for dirpath, _, fnames in os.walk(img_dir):
		concept_str = dirpath.replace(img_dir, "")[1:].split("\\")[0]
		concept = Concept.get_concept_by_str(concept_str)
		dir_imgs = []

		for fname in fnames:
			file_path = os.path.join(dirpath, fname)
			dir_imgs.append(tuple((concept, file_path)))
		random.shuffle(dir_imgs)
		pivot = int(len(dir_imgs)*training_percentage)
		training_data += dir_imgs[:pivot]
		testing_data += dir_imgs[pivot:]

	return training_data, testing_data

