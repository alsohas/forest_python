from typing import Dict, List

from core.graph_members import Node
from core.roadnetwork import RoadNetwork
from region_historic.historic_region import HistoricRegion


class PredictiveNode:
    road_network: RoadNetwork
    predictive_regions: Dict[int, Dict[int, List['PredictiveNode']]]
    region: HistoricRegion
    parent: Node
    root: Node
    depth: int
    max_depth: int
    
    @property
    def level(self):
        return self.max_depth - self.depth

    def __init__(self, rn: RoadNetwork, root: Node, parent:Node,  depth: int, max_depth: int,
                 predictive_regions: Dict[int, Dict[int, List['PredictiveNode']]], historic_region: HistoricRegion):
        self.road_network = rn
        self.root = root
        self.parent = parent
        self.depth = depth
        self. max_depth = max_depth
        self.predictive_regions = predictive_regions
        self.region = historic_region
        self.add_region_reference()

    def add_region_reference(self):
        if self.level not in self.predictive_regions:
            region: Dict[int, List[PredictiveNode]] = {}
            self.predictive_regions[self.level] = region
        region = self.predictive_regions.get(self.level)

        if self.root.node_id not in region:
            node_list: List[PredictiveNode] = []
            region[self.root.node_id] = node_list
        node_list = region.get(self.root.node_id)
        node_list.append(self)

    def expand(self) -> None:
        if self.depth == 0 or self.level == self.max_depth: return
        for n, e in self.root.out_edges.items():
            if n.node_id in self.region.obsolete_nodes: continue
            if self.parent and self.parent.node_id == n.node_id: continue

            child = PredictiveNode(self.road_network, n, self.root, self.depth-1, self.max_depth,
                                   self.predictive_regions, self.region)
            child.expand()
