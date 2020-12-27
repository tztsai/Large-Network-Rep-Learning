""" Graph Utilities """

import logging
import random
import numpy as np
from time import time
from collections import defaultdict

graph_logger = logging.getLogger('Graph')

logging.basicConfig(level=logging.DEBUG)


class Graph:

    def __init__(self, edges, directed=False, weighted=False):
        self.directed = directed
        self.weighted = weighted

        encode = {}  # encode nodes from 0 to #nodes-1
        graph = {}  # graph dict

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
                    if self.weighted:
                        graph[i] = defaultdict(float)
                    else:
                        graph[i] = set()

            i, j = encode[u], encode[v]

            if weighted:
                graph[i][j] += w
                if not self.directed:
                    graph[j][i] += w
            else:
                graph[i].add(j)
                if not self.directed:
                    graph[j].add(i)

        self.num_nodes = len(encode)
        self.nodes = range(self.num_nodes)

        self.adjacency = self.__compute_adjacency(graph)

        self.num_edges = np.count_nonzero(self.adjacency)
        if not self.directed: self.num_edges //= 2

        self.context = {v: tuple(ctx) for v, ctx in graph.items()}

        graph_logger.debug('Constructed a%s %s graph (V=%d, E=%d)'
                           % (' directed' if directed else 'n undirected',
                              'weighted' if weighted else 'unweighted',
                              self.num_nodes, self.num_edges))

    def __compute_adjacency(self, graph):
        A = np.zeros((self.num_nodes, self.num_nodes))
        for i in graph:
            for j in graph[i]:
                A[i, j] = graph[i][j] if self.weighted else 1
        return A


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
    print(g.adjacency)

    G = read_graph('small_sample.txt')
