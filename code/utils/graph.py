""" Graph Utilities """

import logging
import random
import numpy as np
from time import time

graph_logger = logging.getLogger('Graph')
logging.basicConfig(level=logging.DEBUG)


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

        graph_logger.debug('Constructed a%s graph (V=%d, E=%d).'
                           % (' directed' if directed else 'n undirected',
                              self.num_nodes, self.num_edges))

    def __getitem__(self, idx):
        if type(idx) is int:
            return self.neighbors[idx]
        elif type(idx) is tuple and len(idx) == 2:
            i, j = idx
            if j in self.neighbors[i]:
                return self.neighbors[i][j]
            else:
                return 0
        else:
            raise TypeError('invalid index type')
        
    def rand_neighbor(self, node, pi=None):
        """
        Sample a random neighbor of the node.

        Args:
            node: the starting node
            pi (list): the sampling probability distribution

        Returns:
            a random neighbor node
        """
        neighbors = list(self[node])
        pi = np.array(pi) / np.sum(pi)  # normalized array
        return np.random.choice(neighbors, p=pi)
    
    def second_order_bias(self, e, p, q):
        """
        The 2nd order search bias used in node2vec.

        Args:
            e (2-tuple): the previous traversed edge
            p: "walk back" parameter
            q: "walk away" parameter
        """
        u, v = e
        for x in self[v]:
            if x == u:  # walk back
                return 1/p
            elif v in self[u]:
                return 1
            else:
                return 1/q
        
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
    graph_logger.debug('Successfully read graph from "%s" (time: %.2fms).'
                       % (filename, (t1 - t0) * 1000))
    return graph


if __name__ == "__main__":
    edges = [(1, 2), (2, 3), (3, 1), (2, 4), (2, 1), (3, 4)]
    g = Graph(edges, directed=True)
    print(g.neighbors)
    print(g.to_array())

    G = read_graph('../sample_data.txt')
