import json
import random

concept_map_reversed = {
    'Stop': 0,
    'Vorfahrt gewahren': 1,
    'Vorfahrt von rechts': 2,
    'Fahrtrichtung rechts': 3,
    'Fahrtrichtung links': 4,
    'Vorfahrtsstrasse': 5
}

concept_map_reversed2 = {
    'STOP': 0,
    'VORFAHRT_GEWAEHREN': 1,
    'VORFAHRT_VON_RECHTS': 2,
    'VORFAHRT_RECHTS': 2,
    'FAHRTRICHTUNG_RECHTS': 3,
    'FAHRTRICHTUNG_LINKS': 4,
    'VORFAHRTSSTRASSE': 5,
    'VORFAHRTSSTRAßE': 5
}

def load_json_file(filename: str) -> json:
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def read_json_file_and_extract_vectors(filepath: str):
    all_vectors = []
    data = load_json_file(filepath)
    vector2list(all_vectors, data)
    return all_vectors

def vector2list(all_vectors, data):
    for sign in data:
        vector = sign.get('featureVector', '')

        if any(value != 0 for value in vector):
            sign_name = sign.get('classification', '')
            sign_id = concept_map_reversed2.get(sign_name, -1)
            all_vectors.append((sign_id, vector))



def get_training_testing_data_list(data, training_percentage: float):
    random.shuffle(data)

    # Berechne die Grenze für den Trainingsanteil
    pivot = int(len(data) * training_percentage)

    # Splitte die Daten in Trainings- und Testdaten
    training_data = data[:pivot]
    testing_data = data[pivot:]

    return training_data, testing_data