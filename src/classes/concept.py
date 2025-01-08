from enum import Enum, auto

class Concept(Enum):
	VORFAHRT_GEWAEHREN = 0
	VORFAHRT_STRASSE = 1
	STOP = 2
	RECHTS_ABBIEGEN = 3
	LINKS_ABBIEGEN = 4
	RECHTS_VOR_LINKS = 5


	def get_concept_by_str(name: str):
		match name:
			case 'VORFAHRT_GEWAEHREN':
				return Concept.VORFAHRT_GEWAEHREN
			case 'VORFAHRT_STRASSE':
				return Concept.VORFAHRT_STRASSE
			case 'STOP':
				return Concept.STOP
			case 'RECHTS_ABBIEGEN': 
				return Concept.RECHTS_ABBIEGEN
			case 'LINKS_ABBIEGEN':
				return Concept.LINKS_ABBIEGEN
			case 'RECHTS_VOR_LINKS':
				return Concept.RECHTS_VOR_LINKS