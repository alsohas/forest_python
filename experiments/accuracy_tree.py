import uuid
import random
from concurrent.futures.process import ProcessPoolExecutor

import numpy
import pandas
from tqdm import tqdm

from core.graph_members import Node
from core.roadnetwork import RoadNetwork
from forest.predictive_forest import PredictiveForest
from utils.loaders import get_valid_files, data_path, load_dtree
import os
import json
from joblib import parallel_backend, delayed, Parallel

ROAD_NETWORK = RoadNetwork()


def eval_trip(trip, dtree, step, radius, pbar):
    count = 0
    correct = 0
    if len(trip) < step + 1: return count, correct
    forest = PredictiveForest(ROAD_NETWORK, step, tree=dtree)
    for index, point in enumerate(trip):
        if index + step >= len(trip): break
        node: int = ROAD_NETWORK.reverse_mapped_dictionary.get(str(point))
        actual = ROAD_NETWORK.reverse_mapped_dictionary.get(str(trip[index + step]))
        forest.update(ROAD_NETWORK.nodes.get(node).location, radius)
        try:
            pred, table = forest.get_predicted_node()
        except:
            continue
        if table is None or actual not in table:
            continue
        if pred == 0:
            continue
        count += 1
        if actual == pred:
            correct += 1

    return count, correct


# load all the files
def benchmark(v_file, dtree, step, radius):
    with open(v_file, 'r') as json_file:
        trips = json.load(json_file)
        count = 0
        correct = 0
        pbar = tqdm(trips)
        for trip in pbar:
            _count, _correct = eval_trip(trip, dtree, step, radius, pbar)
            count += _count
            correct += _correct
            pbar.set_description(f'Accuracy: {correct}/{count}')
        pbar.close()
        return count, correct, len(trips)


def gen_benchmark(radius, valid_files, step, dtree):
    results = pandas.DataFrame(columns=['region_size', 'steps', 'trips', 'total', 'correct'])
    count = 0
    correct = 0
    len_trips = 0
    for v_file in random.sample(valid_files, 10):
        _count, _correct, _len_trips = benchmark(v_file, dtree, step, radius)
        count += _count
        correct += _correct
        len_trips += _len_trips

    results.loc[len(results)] = (radius, step, len_trips, count, correct)
    print(results.head(99))
    results.to_csv(f'accuracy_{step}_{radius}m_naive_{uuid.uuid4().hex}.csv', index=False)


def load_trip():
    STEPS = [1, 2, 3, 4, 5]
    RADIUS = [25, 50, 75, 100]
    valid_files = get_valid_files(os.path.join(data_path, 'test'))

    for step in STEPS:
        dtree = None # load_dtree(step)
        with ProcessPoolExecutor(max_workers=4) as worker:
            for radius in RADIUS:
                worker.submit(gen_benchmark, radius, valid_files, step, dtree)
