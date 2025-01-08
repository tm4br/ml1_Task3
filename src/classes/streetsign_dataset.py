
import os
import torch
import cv2

from torch.utils.data import Dataset

from classes.concept import Concept


class StreetsignDataset(Dataset):

	def __init__(self, tensors_concept_list: list[tuple[Concept, torch.Tensor]]) -> None:
		self.data = tensors_concept_list
		self.class_map = {
							Concept.VORFAHRT_GEWAEHREN : Concept.VORFAHRT_GEWAEHREN.value,
							Concept.VORFAHRT_STRASSE: Concept.VORFAHRT_STRASSE.value,
							Concept.STOP: Concept.STOP.value,
							Concept.RECHTS_ABBIEGEN: Concept.RECHTS_ABBIEGEN.value,
							Concept.LINKS_ABBIEGEN: Concept.LINKS_ABBIEGEN.value,
							Concept.RECHTS_VOR_LINKS: Concept.RECHTS_VOR_LINKS.value
						}
		

	def __len__(self) -> int:
		return len(self.data)
	

	def __getitem__(self, index):
		concept = self.data[index][0]
		concept_id = self.class_map[concept]
		concept_id = torch.tensor(concept_id, dtype=torch.long)

		img_tensor = self.data[index][1]
		return img_tensor, concept_id
