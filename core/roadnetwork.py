from typing import Dict, Iterator, List
from utils.dict_generator import data_path
import os
import geopandas as gpd
import numpy as np

from core.coordinate import Coordinate
from core.graph_members import Edge, Node
import json

from utils.dict_generator import load_dictionary
from shapely import wkt

import numba


def get_bbox(x: float, y: float, radius: float):
    """
    Args:
        x: longitude
        y: latitude
        radius: radius in KM
    """
    radius = radius / 1000
    radius_earth = 6378.1
    bearings = np.array([np.radians(0), np.radians(90), np.radians(180), np.radians(270)])

    origin_lat = np.radians(y)
    origin_lng = np.radians(x)
    min_lng = min_lat = max_lng = max_lat = None
    for b in bearings:
        lat = np.arcsin(np.sin(origin_lat) * np.cos(radius / radius_earth) +
                        np.cos(origin_lat) * np.sin(radius / radius_earth) * np.cos(b))
        lng = origin_lng + np.arctan2(np.sin(b) * np.sin(radius / radius_earth) * np.cos(origin_lat),
                                      np.cos(radius / radius_earth) - np.sin(origin_lat) * np.sin(lat))
        lng = np.degrees(lng)
        lat = np.degrees(lat)
        if min_lat is None:
            min_lng = max_lng = lng
            max_lat = min_lat = lat
            continue
        if lat > max_lat: max_lat = lat
        if lat < min_lat: min_lat = lat
        if lng > max_lng: max_lng = lng
        if lng < min_lng: min_lng = lng

    return min_lng, min_lat, max_lng, max_lat


class RoadNetwork:
    reverse_mapped_dictionary: Dict[str, int]
    index_structure: gpd.GeoDataFrame
    nodes: Dict[int, Node]

    def __init__(self):
        self.nodes = {}
        self.reverse_mapped_dictionary = {}
        self.build_network()

    def get_nodes_in_range(self, center: Coordinate, radius: float) -> List[Node]:
        """
        Args:
            center: center
            radius: radius in KM
        """
        min_lng, min_lat, max_lng, max_lat = get_bbox(center.lng, center.lat, radius)
        gdf = self.index_structure.cx[min_lng:max_lng, min_lat:max_lat]
        n_ids_ = gdf['node_id']
        nodes = []
        for n in n_ids_:
            nodes.append(self.nodes.get(n))
        return nodes

    def build_network(self):
        self.reverse_mapped_dictionary = load_dictionary()
        self._initialize_nodes()
        self._initialize_edges()
        self._build_index()
        print("Finished building network")

    def _initialize_nodes(self):
        with open(os.path.join(data_path, 'chengdu_nodes.json'), 'r') as node_file:
            all_nodes = json.load(node_file)

        for n in all_nodes:
            c = n['coordinate']
            n_id_ = str(n['id'])
            if n_id_ not in self.reverse_mapped_dictionary:
                self.reverse_mapped_dictionary[n_id_] = len(self.reverse_mapped_dictionary)
            n_id = self.reverse_mapped_dictionary.get(n_id_, None)
            if not n_id: continue
            node = Node(n_id, c['lng'], c['lat'])
            if node.node_id in self.nodes: continue
            self.nodes[node.node_id] = node
        print(f"Loaded {len(self.nodes)} nodes")

    def _initialize_edges(self):
        with open(os.path.join(data_path, 'chengdu_edges.json'), 'r') as edge_file:
            all_edges = json.load(edge_file)
        for e in all_edges:
            source_id = self.reverse_mapped_dictionary.get(str(e[0]))
            destination_id = self.reverse_mapped_dictionary.get(str(e[1]))
            source = self.nodes.get(source_id, None)
            destination = self.nodes.get(destination_id, None)
            if not source or not destination: continue
            edge = Edge(source, destination)
            destination.add_in_edge(source, edge)
            source.add_out_edge(destination, edge)
        print(f"Loaded {len(all_edges)} edges")

    def _build_index(self):
        points = []
        n_ids = []
        for n_id, node in self.nodes.items():
            wkt_string = f"POINT ({node.location.shape()[0]} {node.location.shape()[1]})"
            point = wkt.loads(wkt_string)
            points.append(point)
            n_ids.append(n_id)
        df_cols = ['node_id', 'coordinate']
        self.index_structure = gpd.GeoDataFrame(columns=df_cols, crs='epsg:4326',
                                                geometry='coordinate', data=list(zip(n_ids, points)))
