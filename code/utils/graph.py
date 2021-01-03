""" Graph Utilities """

import random
import logging
import numpy as np
from time import time
import numpy.random as npr

logger = logging.getLogger('Graph')


class Graph:

    def __init__(self, edges, labels=None, directed=False):
        self.directed = directed
        self.neighbors = {}  # dict of node context
        self.labels = labels

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
            idx = int(idx)
            return self.neighbors[idx]
        except:
            try:
                i, j = idx
                if j in self.neighbors[i]:
                    return self.neighbors[i][j]
                else:
                    return 0
            except:
                raise TypeError('Invalid item type!')
            
    def __contains__(self, obj):
        try:
            obj = int(obj)
            return obj in self.nodes
        except:
            try:
                i, j = obj
                return j in self[i]
            except:
                return False

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
        

def read_graph(graph_file, labels_file=None, multi_labels=False, **graph_type):

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
        
    graph = Graph(edges, labels, **graph_type)
    
    t1 = time()
    logger.debug('Successfully read graph from "%s"%s (time: %dms).' % 
                 (graph_file, f' and labels from "{labels_file}"', (t1 - t0) * 1000))
    return graph


if __name__ == "__main__":
    G = read_graph('code/datasets/cocit/data_CoCit_CoCit-edgelist-unitweight.txt',
                   'code/datasets/cocit/data_CoCit_CoCit-labels.txt')