import random
import numpy as np
import torch as tc
from utils.graph import Graph, read_graph
import config


class DeepWalk:
    ws = 3      # window size
    wl = 5      # walk length

    def __init__(self, graph: Graph):
        self.G = graph
        self.N = graph.num_nodes
        self.D = config.D
        self.Z = np.random.rand((self.N, self.D))

    def sample_walk(self):
        """ Generate a sample of random walks for each node. """
        def walk(v):
            seq = [v]
            for _ in range(self.wl):
                v = self.G.rand_neighbor(v)
                seq.append(v)
            return seq

        return map(walk, self.G.nodes)

    def sample_context_graph(self):
        edges = []
        for w in self.sample_walk():
            for i, u in enumerate(w):
                j1 = max(0, i - self.ws)
                j2 = min(len(w), i + self.ws + 1)
                for j in range(j1, j2):
                    if i == j: continue
                    v = w[j]
                    # u: center node, v: context node
                    edges.append([u, v])
        random.shuffle(edges)
        return np.array(edges)

    def train(self, epochs=10):
        CG = self.sample_context_graph()

    def deep_walk(self):
        for _ in range(self.epochs):
            walks = self.sample_walks()
            random.shuffle(walks)
            self.skip_gram(walks)


class SkipGram:
    pass


if __name__ == "__main__":
    g = read_graph('small.txt')
    dw = DeepWalk(g)
