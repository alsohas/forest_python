import gzip
import json
import os
import pickle
from typing import List, Dict

from sklearn.tree import DecisionTreeClassifier

base_path = '/run/media/ai/22d5a295-a1e8-4720-98e3-fb98255a8ecf/forest_python/'
data_path = os.path.join(base_path, 'data')
pickle_folder = os.path.join(data_path, 'pickles')
trips_file_path = os.path.join(data_path, 'map_matched_all/')
mapped_dict_file_name = 'mapped_dict.json'
nodes_file_name = 'chengdu_nodes.json'
weights_file_name = 'weights.json'


def get_valid_files(folder) -> List[str]:
    """
    walks a data folder and subdirectories to return all data files
    """
    gps_folders = os.listdir(folder)
    valid_files = []
    for gps_folder in gps_folders:
        gps_folder_path = os.path.join(folder, gps_folder)
        all_files = os.listdir(gps_folder_path)
        for f in all_files:
            f_path = os.path.join(gps_folder_path, f)
            valid_files.append(f_path)
    return valid_files


def load_dictionary() -> Dict[str, int]:
    with open(os.path.join(data_path, mapped_dict_file_name), 'r') as mapped_dict:
        return json.load(mapped_dict)


def load_weights() -> Dict[int, int]:
    with open(os.path.join(data_path, weights_file_name), 'r') as mapped_dict:
        return json.load(mapped_dict)


def load_dtree(step: int) -> DecisionTreeClassifier:
    fp = gzip.open(os.path.join(pickle_folder, f'dtree_{step}.pkl'), 'rb')
    dtree = pickle.load(fp)
    fp.close()
    return dtree


