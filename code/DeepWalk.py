import random
import numpy as np
import torch
from torch.nn.functional import log_softmax
from utils.graph import Graph, read_graph
from utils.tree import HuffmanTree
import config


class DeepWalk:
    wl = 5      # walk length
    ws = 3  # window size

    def __init__(self, graph: Graph):
        self.G = graph
        self.N = graph.num_nodes
        self.T = HuffmanTree()
        self.T.build_from_graph(graph)
        self.D = config.D
        self.Z = np.random.rand((self.N, self.D))

    def sample_walk(self):
        """Generate a sample of random walks for each node."""
        def walk(v):
            seq = [v]
            for _ in range(self.wl):
                v = self.G.rand_neighbor(v)
                seq.append(v)
            return seq

        return map(walk, self.G.nodes)

    def context_graph(self, walks):
        """Generate a context graph from sampled walks."""
        edges = []
        for w in walks:
            for i in range(len(w)):
                j1 = max(0, i - self.ws)
                j2 = min(len(w), i + self.ws + 1)
                for j in range(j1, j2):
                    # i: center node, j: context node
                    if i == j: continue
                    edges.append([w[i], w[j]])
        graph = np.array(edges)
        np.random.shuffle(graph)
        return graph

    def train(self, epochs=10):
        for epoch in range(epochs):
            walks = self.sample_walk()
            ctx = self.context_graph(walks)


if __name__ == "__main__":
    g = read_graph('small.txt')
    dw = DeepWalk(g)
