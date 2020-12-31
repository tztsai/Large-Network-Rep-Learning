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
        if pi:
            pi = np.array(pi) / np.sum(pi)  # normalized array
            return np.random.choice(neighbors, p=pi)
        else:
            return np.random.choice(neighbors)
        
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

def alias_setup(probs):
    '''
    this code was taken from this page:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    K       = len(probs)
    q       = np.zeros(K)
    J       = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    this code was taken from this page:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    K  = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand()*K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]


if __name__ == "__main__":
    edges = [(1, 2), (2, 3), (3, 1), (2, 4), (2, 1), (3, 4)]
    g = Graph(edges, directed=True)
    print(g.neighbors)
    print(g.to_array())

    G = read_graph('small.txt')
