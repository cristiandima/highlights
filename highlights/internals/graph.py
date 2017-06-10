from collections import defaultdict

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.neighbours = defaultdict(set)

    def add_node(self, key, value=None):
        """ Adds a node to the graph
        Args:
            key: node unique id
            value: optional value associated with the node
        """
        self.nodes[key] = value

    def add_edge(self, edge, weight=1):
        """ Connects two nodes
        Args:
            edge: tuple of two node keys
            weight: value associated with the edge
        """
        self.edges[edge] = weight
        self.neighbours[edge[0]].add(edge[1])
        self.neighbours[edge[1]].add(edge[0])

    def remove_dangling(self):
        for node in list(self.nodes.keys()):
            if len(self.neighbours[node]) == 0:
                del self.neighbours[node]
                del self.nodes[node]
