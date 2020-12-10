import json
import os
from tqdm import tqdm

base_path = '/run/media/ai/22d5a295-a1e8-4720-98e3-fb98255a8ecf/forest_python'
data_path = os.path.join(base_path, 'data')
trips_file_path = os.path.join(data_path, 'map_matched_all/')
mapped_dict_file_name = 'mapped_dict.json'
nodes_file_name = 'chengdu_nodes.json'


def load_dictionary():
    with open(os.path.join(data_path, mapped_dict_file_name), 'r') as mapped_dict:
        return json.load(mapped_dict)


def generate_dictionary():
    reverse_map = {0: 0}
    nodes_file_path = os.path.join(data_path, nodes_file_name)

    with open(nodes_file_path, 'r') as node_file:
        all_nodes = json.load(node_file)
    for n in all_nodes:
        n_id = int(n['id'])
        if n_id in reverse_map: continue
        reverse_map[int(n['id'])] = len(reverse_map)

    all_gps_folders = os.listdir(trips_file_path)
    all_trips_files = []
    for g_folder in all_gps_folders:
        if 'gps' not in g_folder: continue
        folder_path = os.path.join(trips_file_path, g_folder)
        trip_files = os.listdir(folder_path)
        for trip_file in trip_files:
            file_path = os.path.join(folder_path, trip_file)
            all_trips_files.append(file_path)

    pbar = tqdm(total=len(all_trips_files), desc="Reading trip files")
    for tf in all_trips_files:
        with open(tf, 'r') as trip_file:
            trips = json.load(trip_file)
        for trip in trips:
            for p in trip:
                n_id = int(p)
                if n_id in reverse_map: continue
                reverse_map[n_id] = len(reverse_map)
        pbar.update(1)

    with open(os.path.join(data_path, mapped_dict_file_name), 'w') as mapped_dict:
        json.dump(reverse_map, mapped_dict, indent=2)
