""" Graph Utilities """

import logging
import numpy as np
from time import time
import numpy.random as npr

logger = logging.getLogger('Graph')


class Graph:

    def __init__(self, edges, directed=False):
        self.directed = directed
        self.neighbors = {}  # dict of node context

        encode = {}  # encode nodes from 0 to #nodes-1

        for edge in edges:
            try:
                if len(edge) == 2:
                    u, v = edge
                    w = 1
                else:
                    u, v, w = edge
            except (TypeError, ValueError):
                raise ValueError('Invalid form of edges!')

            for n in [u, v]:
                if n not in encode:
                    i = encode[n] = len(encode)
                    self.neighbors[i] = {}

            i, j = encode[u], encode[v]

            self.neighbors[i][j] = self.neighbors[i].get(j, 0) + w
            if not self.directed:
                self.neighbors[j][i] = self.neighbors[j].get(i, 0) + w

        self.num_nodes = len(encode)
        self.nodes = range(self.num_nodes)

        self.num_edges = sum(len(nbs) for nbs in self.neighbors.values())
        if not self.directed: self.num_edges //= 2

        self.decode = {i: v for v, i in encode.items()}

        logger.debug(f"Constructed a{' directed' if directed else 'n undirected'}"
                     f" graph (V={self.num_nodes}, E={self.num_edges}).")

    def __getitem__(self, idx):
        try:
            return self.neighbors[idx]
        except (IndexError, TypeError):
            if type(idx) is tuple and len(idx) == 2:
                i, j = idx
                if j in self.neighbors[i]:
                    return self.neighbors[i][j]
                else:
                    return 0
            else: raise

    def weight(self, u, v=None):
        """The weight of a node or an edge, depending on the number of arguments."""
        if v is None:
            return sum(self[u].values())
        else:
            return self[u, v]
        
    def sample_neighbors(self, node, k=1):
        """
        Generate a sample of neighbors of the node.

        Args:
            node: the node from whose neighborhood the sample is drawn
            k (default 1): the sample size

        Returns:
            a random neighbor if k = 1; otherwise a list of sampled neighbors
        """
        neighbors = list(self[node])
        sample = random.sample(neighbors, k)
        return sample[0] if k == 1 else sample
        
    def to_array(self):
        return np.array([[self[i, j] for j in range(self.num_nodes)]
                         for i in range(self.num_nodes)])


def read_graph(filename, **graph_type):
    t0 = time()
    with open(filename, 'r') as f:
        edges = ([int(s) for s in line.split()]
                 for line in f.readlines())
    graph = Graph(edges, **graph_type)
    t1 = time()
    logger.debug('Successfully read graph from "%s" (time: %dms).'
                       % (filename, (t1 - t0) * 1000))
    return graph


if __name__ == "__main__":
    edges = [(1, 2), (2, 3), (3, 1), (2, 4), (2, 1), (3, 4)]
    g = Graph(edges, directed=True)
    print(g.neighbors)
    print(g.to_array())

    G = read_graph('small.txt')
