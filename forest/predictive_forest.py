from typing import Dict, List, Set, Union

import keras
from sklearn.tree import DecisionTreeClassifier

from core.coordinate import Coordinate
from core.roadnetwork import RoadNetwork
from region_historic.historic_node import HistoricNode
from region_historic.historic_region import HistoricRegion
from region_predictive.predictive_node import PredictiveNode
import numpy as np

OFFSET = 8


class PredictiveForest:
    current_step: int
    depth: int
    predictive_regions: Dict[int, Dict[int, List[PredictiveNode]]]
    road_network: RoadNetwork
    historic_tree: HistoricRegion
    tree: DecisionTreeClassifier

    def __init__(self, rn: RoadNetwork, depth: int, tree: DecisionTreeClassifier = None):
        self.road_network = rn
        self.depth = depth
        self.current_step = 0
        self.historic_tree = HistoricRegion()
        self.tree = tree

    def update(self, center: Coordinate, radius: float):
        if not self.current_step:
            self.historic_tree.update(self.road_network.get_nodes_in_range(center, radius))
            region = self.historic_tree.regions.get(self.current_step)
            self.expand_trees(region)
            self.current_step += 1
            return

        past_nodes = self.historic_tree.regions.get(self.current_step - 1)
        children: Set[int] = set()
        for n_id, node in past_nodes.items():
            children = children.union(node.children)

        current_nodes: Set[int] = set()
        for node in self.road_network.get_nodes_in_range(center, radius):
            current_nodes.add(node.node_id)

        current_nodes = current_nodes.intersection(children)

        obsolete_parents: Set[int] = set()
        valid_parents: Set[int] = set()
        for n_id, past_node in past_nodes.items():
            past_node.children = past_node.children.intersection(current_nodes)
            if not len(past_node.children):
                obsolete_parents.add(n_id)
                continue
            valid_parents.add(n_id)

        new_region: Dict[int, HistoricNode] = {}
        for n_id in current_nodes:
            node = self.road_network.nodes.get(n_id)
            current_node = HistoricNode(node, valid_parents)
            new_region[n_id] = current_node
        self.historic_tree.regions[self.current_step] = new_region
        self.prune_regions(self.current_step - 1, obsolete_parents)
        region = self.historic_tree.regions.get(self.current_step)
        self.expand_trees(region)
        self.current_step += 1

    def expand_trees(self, new_region: Dict[int, HistoricNode]):
        self.predictive_regions: Dict[int, Dict[int, List[PredictiveNode]]] = {}
        for n_id, node in new_region.items():
            if n_id in self.historic_tree.obsolete_nodes: continue
            node = self.road_network.nodes.get(n_id)
            predictive_node = PredictiveNode(self.road_network, node, None, self.depth, self.depth,
                                             self.predictive_regions, self.historic_tree)
            predictive_node.expand()

    def prune_regions(self, steps: int, obsolete_nodes: Set[int]):
        self.historic_tree.obsolete_nodes = self.historic_tree.obsolete_nodes.union(obsolete_nodes)
        if not len(obsolete_nodes): return

        obsolete_parents: Set[int] = set()
        region = self.historic_tree.regions.get(steps)
        parental_region = self.historic_tree.regions.get(steps - 1)
        for n_id in obsolete_nodes:
            node = region.pop(n_id)
            parents = node.parents
            if not parents: continue
            for p_id in parents:
                parent = parental_region.get(p_id)
                parent.children.remove(n_id)
                if not len(parent.children): obsolete_parents.add(p_id)
        return self.prune_regions(steps - 1, obsolete_parents)

    def get_predicted_node(self):
        if self.tree is not None:
            possible_trajectories = self._get_trajectories()
            return self._predict_tree(possible_trajectories)
        return self._predict_naive()

    def _predict_naive(self):
        last_region = self.predictive_regions.get(self.depth)
        max_node = list(last_region.keys())[0]
        for n_id, node_list in last_region.items():
            if len(node_list) > len(last_region.get(max_node, [])):
                max_node = n_id
        return max_node, last_region.keys()



    def _get_trajectories(self):
        trajectories: List[List[int]] = self._get_trajectories_helper(self.historic_tree.region_count - 1)
        assert len(trajectories) >= 1
        assert len(trajectories[0]) == self.historic_tree.region_count
        t_list = keras.preprocessing.sequence.pad_sequences(trajectories, maxlen=OFFSET, dtype='int16')
        return t_list

    def _get_trajectories_helper(self, region_index: int):
        trajectories: List[List[int]] = []
        if region_index < 0:
            return trajectories
        _trajectories = self._get_trajectories_helper(region_index - 1)
        possible_nodes = self.historic_tree.regions.get(region_index)

        if not len(_trajectories):
            for n_id in possible_nodes.keys():
                trajectories.append([n_id])
            return trajectories
        for trajectory in _trajectories:
            # because the list is in order of traversal,
            # last node is parent to something in current region
            p_id = trajectory[-1]
            for n_id, node in possible_nodes.items():
                if p_id not in node.parents: continue
                new_trajectory = trajectory + [n_id]
                trajectories.append(new_trajectory)

        return trajectories

    def _predict_tree(self, possible_trajectories):
        try:
            probabilities = self.tree.predict_proba(possible_trajectories)
        except Exception:
            return 0

        sum_prob = None
        for p in probabilities:
            if sum_prob is None:
                sum_prob = np.add(p, 0)
            else:
                sum_prob = np.add(sum_prob, p)

        possible_nodes = list(self.predictive_regions.get(self.depth, {}).keys())
        cond = None
        for p in possible_nodes:
            if cond is None:
                cond = (p == self.tree.classes_)
            else:
                cond |= (p == self.tree.classes_)
        if not possible_nodes:
            return self.tree.classes_[np.argmax(sum_prob)], None
        predicted_classes_idx = np.where(cond)
        predicted_classes = np.take(self.tree.classes_, predicted_classes_idx)
        predicted_probabilities = np.take(sum_prob, predicted_classes_idx)
        max_prob_idx = np.argmax(predicted_probabilities)
        predicted_class_idx = predicted_classes_idx[0][max_prob_idx]
        return self.tree.classes_[predicted_class_idx], predicted_classes[0].tolist()

