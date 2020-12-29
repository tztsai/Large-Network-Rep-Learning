import random
import numpy as np
import torch

from utils.graph import Graph, read_graph
from utils.huffman import HuffmanTree
from utils.funcs import sigmoid
import config


class DeepWalk:
    wl = 5  # walk length
    ws = 3  # window size
    alpha = config.ALPHA

    def __init__(self, graph: Graph):
        self.G = graph
        self.N = graph.num_nodes
        self.D = config.D

        # build a Huffman binary tree from graph nodes
        node_weights = [sum(graph.neighbors[n].values())
                        for n in graph.nodes]
        self.T = HuffmanTree(node_weights)

        # latent representation of nodes
        self.Z1 = torch.empty((self.N, self.D), requires_grad=True)
        # latent representation of inner nodes of the tree
        self.Z2 = torch.empty((self.N-1, self.D), requires_grad=True)

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
        return edges

    def train(self, epochs=10):
        for epoch in range(epochs):
            walks = self.sample_walk()
            ctx = self.context_graph(walks)
            loss = self.loss(ctx)

            loss.backward()
            with torch.no_grad():
                self.Z1 -= self.Z1.grad * self.alpha
                self.Z2 -= self.Z2.grad * self.alpha
                self.Z1.grad.zero_()
                self.Z2.grad.zero_()

    def log_softmax(self, u, v):
        """log p(u|v) where p is the hierarchical softmax function"""
        lp = 0  # log probability
        n = u
        while True:
            p = self.T.parent[n]
            if p < 0: break
            s = 1 - self.T.code[n] * 2
            x = torch.dot(self.Z1[v], self.Z2[p-self.N])
            lp += np.log(sigmoid(s * x))
        return lp

    def loss(self, ctx):
        ls = torch.tensor([self.log_softmax(v, u) for u, v in ctx])
        return -torch.sum(ls)

    def similarity(self, u, v):
        with torch.no_grad():
            Zu = self.Z1[u]
            Zv = self.Z1[v]
            Zu /= np.linalg.norm(Zu)
            Zv /= np.linalg.norm(Zv)
            return Zu @ Zv


if __name__ == "__main__":
    g = read_graph('utils/small.txt')
    dw = DeepWalk(g)
    dw.train()
