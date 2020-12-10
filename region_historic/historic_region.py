from typing import Set, Dict, Iterable, List

from core.graph_members import Node
from region_historic.historic_node import HistoricNode


class HistoricRegion:
    obsolete_nodes: Set[int]
    regions: Dict[int, Dict[int, HistoricNode]]

    def __init__(self):
        self.obsolete_nodes = set()
        self.regions = {}

    @property
    def region_count(self):
        return len(self.regions)

    def update(self, nodes: List[Node]):
        new_region: Dict[int, HistoricNode] = {}
        for n in nodes:
            historic_node = HistoricNode(n, set())
            new_region[n.node_id] = historic_node
        self.regions[self.region_count] = new_region
