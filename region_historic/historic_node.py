from core.graph_members import Node
from typing import Set


class HistoricNode:
    node: Node
    children: Set[int]
    parents: Set[int]

    def __init__(self, node: Node, parents: Set[int]):
        self.node = node
        self.children = set()
        self.parents = set()
        self._add_parents(node, parents)
        self._add_children(node)

    def _add_parents(self, node: Node, parents: Set[int]):
        if not len(parents):
            return
        for n, e in node.in_edges.items():
            if n.node_id in parents:
                self.parents.add(n.node_id)

    def _add_children(self, node):
        for n, e in node.out_edges.items():
            if node.node_id in self.parents: continue
            self.children.add(n.node_id)
