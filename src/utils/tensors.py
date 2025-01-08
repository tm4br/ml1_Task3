import os
import cv2
import time
import torch

from os.path import abspath, join, dirname

from classes.concept import Concept


def create_and_save_tensors(img_dir_path: str, tensors_dir: str, img_dim: tuple = (25, 25)):
	concept_tensors = []
	last_concept = ""
	for dirpath, _, fnames in os.walk(img_dir_path):
		if not fnames:
			continue

		concept = dirpath.replace(img_dir_path, "")[1:].split("\\")[0]
		print(concept)

		if last_concept != concept and len(last_concept) > 0:
			# if new concept: save tensors of old concept
			save_path = os.path.join(tensors_dir, f"{last_concept}.pt")
			torch.save(concept_tensors, save_path)
			last_concept = concept
			concept_tensors = []
		elif not last_concept:
			last_concept = concept

		for fname in fnames:
			img_path = os.path.join(dirpath, fname)
			img = cv2.imread(img_path)
			img = cv2.resize(img, img_dim)
			img = img / 255.0	# normalize pixels to range 0..1
			img_tensor = torch.from_numpy(img).float()
			img_tensor = img_tensor.permute(2, 0, 1)
			
			concept_tensors.append(img_tensor)
			
	save_path = os.path.join(tensors_dir, f"{last_concept}.pt")
	torch.save(concept_tensors, save_path)


def load_image_tensors(tensor_dir_path: str) -> dict[Concept, list[torch.Tensor]]:
	out = {}
	for dirpath, _, fnames in os.walk(tensor_dir_path):
		if not fnames:
			continue
		
		for fname in fnames:
			concept_str = fname.removesuffix(".pt")
			concept = Concept.get_concept_by_str(concept_str)
			load_path = os.path.join(dirpath, fname)
			tensors: list[torch.Tensor] = torch.load(load_path)
			out.update({concept: tensors})
	return out