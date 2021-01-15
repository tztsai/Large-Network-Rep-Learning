""" Graph Utilities """

import random
import logging
import numpy as np
from time import time
from .sampling import alias
from .funcs import timer

logger = logging.getLogger('Graph')
random.seed(1)


class Graph:

    def __init__(self, edges, labels=None, directed=False):
        self.neighbors = {}
        self.weights = {}
        self.labels = labels
        self.directed = directed

        encode = {}  # encode nodes from 0 to #nodes-1
        self.decode = {}

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
                    self.weights[i] = {}
                    self.decode[i] = n

            i, j = encode[u], encode[v]

            self.weights[i][j] = self.weights[i].get(j, 0) + w
            if not self.directed:
                self.weights[j][i] = self.weights[j].get(i, 0) + w
                
        self.num_nodes = len(encode)
        self.nodes = range(self.num_nodes)
        
        self.num_edges = 0
        
        for u in self.nodes:
            nbs = tuple(self.weights[u])
            self.neighbors[u] = nbs
            self.num_edges += len(nbs)
        
        if not self.directed:
            self.num_edges //= 2

        self._neg_sampler = None

        logger.debug(f"Constructed a{' directed' if directed else 'n undirected'}"
                     f" graph (V={self.num_nodes}, E={self.num_edges}).")

    def __getitem__(self, node):
        u = int(node)
        return self.neighbors[u]
            
    def __contains__(self, obj):
        try:
            u = int(obj)
            return u in self.nodes
        except:
            try:
                u, v = obj
                return v in self[u]
            except:
                return False

    def weight(self, u, v=None):
        """The weight of a node or an edge, depending on the number of arguments."""
        if v is None:
            return sum(self.weights[u].values())
        else:
            return self.weights[u][v]
        
    def sample_neighbors(self, node, k=1):
        """
        Generate a sample of neighbors of the node.

        Args:
            node: the node from whose neighborhood the sample is drawn
            k (default 1): the sample size

        Returns:
            a random neighbor if k = 1; otherwise a list of sampled neighbors
        """
        neighbors = self[node]
        sample = [random.choice(neighbors) for _ in range(k)]
        return sample[0] if k == 1 else sample

    def noise_sample(self):
        if self._neg_sampler is None:
            # init negative sampler
            node_weights = np.array([self.weight(u) for u in self.nodes], dtype=np.float)
            node_weights /= sum(node_weights)  # normalize
            self._neg_sampler = alias(node_weights ** 0.75)

        return self._neg_sampler.draw()


def read_graph(graph_file, labels_file=None, multi_labels=False, directed=False):

    def read_edge(line):
        return list(map(int, line.split()))

    def read_label(line):
        tokens = list(map(int, line.split()))
        if multi_labels:
            node, *label = tokens
        else:
            node, label = tokens
        return node, label
    
    t0 = time()
    
    with open(graph_file, 'r') as f:
        edges = map(read_edge, f.readlines())
        
    if labels_file:
        with open(labels_file, 'r') as f:
            labels = dict(map(read_label, f.readlines()))
    else:
        labels = None
        
    graph = Graph(edges, labels, directed)
    
    logger.debug('Successfully read graph from "%s"%s.' % 
                 (graph_file, f' and labels from "{labels_file}"' if labels_file else ''))
    return graph


if __name__ == "__main__":
    G = read_graph('code/datasets/cocit/data_CoCit_CoCit-edgelist-unitweight.txt',
                   'code/datasets/cocit/data_CoCit_CoCit-labels.txt')
