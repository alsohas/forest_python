from typing import Dict

from core.coordinate import Coordinate


class Edge:
    destination: 'Node'
    source: 'Node'

    def __init__(self, source: 'Node', destination: 'Node'):
        self.destination = destination
        self.source = source

    def __hash__(self):
        return hash(f"{self.destination}_{self.source}")


class Node:
    out_edges: Dict['Node', Edge]
    in_edges: Dict['Node', Edge]

    def __init__(self, node_id: int, lng: float, lat: float):
        self.node_id = node_id
        self.location = Coordinate(lng, lat)
        self.out_edges = {}
        self.in_edges = {}

    def add_out_edge(self, node: 'Node', edge: Edge):
        if node not in self.out_edges: self.out_edges[node] = edge
        if node not in self.in_edges: self.in_edges[node] = edge

    def add_in_edge(self, node: 'Node', edge: Edge):
        if node not in self.out_edges: self.out_edges[node] = edge
        if node not in self.in_edges: self.in_edges[node] = edge

    def __hash__(self):
        return hash(self.node_id)

    def __repr__(self):
        return str(self.node_id)
