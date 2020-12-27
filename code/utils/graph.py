""" Graph Utilities """

import logging
import random
import numpy as np
import scipy as sp
from scipy.sparse import issparse, csr_matrix
from time import time
from collections import defaultdict

graph_logger = logging.getLogger('Graph')

logging.basicConfig(level=logging.DEBUG)


class Graph:

    def __init__(self, edges, directed=False):
        self.directed = directed
        self.context = {}   # dict of node context

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
                    self.context[i] = defaultdict(float)

            i, j = encode[u], encode[v]

            self.context[i][j] += w
            if not self.directed:
                self.context[j][i] += w

        self.num_nodes = len(encode)
        self.nodes = range(self.num_nodes)

        self.num_edges = sum(len(ctx) for ctx in self.context.values())
        if not self.directed: self.num_edges //= 2

        graph_logger.debug('Constructed a%s graph (V=%d, E=%d)'
                           % (' directed' if directed else 'n undirected',
                              self.num_nodes, self.num_edges))

    def __getitem__(self, idx):
        if type(idx) is int:
            return self.context[idx]
        elif type(idx) is tuple and len(idx) == 2:
            i, j = idx
            if j in self.context[i]:
                return self.context[i][j]
            else:
                return 0
        else:
            raise TypeError('invalid index type')
        
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
    graph_logger.debug('Successfully read graph from "%s". Time: %.2fms'
                       % (filename, (t1 - t0) * 1000))
    return graph


if __name__ == "__main__":
    edges = [(1, 2), (2, 3), (3, 1), (2, 4), (2, 1), (3, 4)]
    g = Graph(edges)
    print(g.to_array())

    G = read_graph('small_sample.txt')