import random
import logging
import numpy as np
import torch
from time import time

from utils.graph import Graph, read_graph
from utils.huffman import HuffmanTree
from utils.funcs import pbar, pmap
import config

logger = logging.getLogger('DeepWalk')


class DeepWalk:
    wl = 6                  # walk length
    ws = 2                  # window size
    bs = 128                # batch size
    lr = config.ALPHA       # learning rate
    tau = 0.95              # exponential annealing rate

    def __init__(self, graph: Graph):
        self.G = graph
        self.N = graph.num_nodes
        self.D = config.D

        # build a Huffman binary tree from graph nodes
        node_weights = [sum(graph.neighbors[n].values())
                        for n in graph.nodes]
        self.T = HuffmanTree(node_weights)

        # latent representation of nodes
        self.Z1 = torch.rand(self.N, self.D, requires_grad=True)
        # latent representation of inner nodes of the tree
        self.Z2 = torch.rand(self.N-1, self.D, requires_grad=True)

    def sample_walks(self):
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
        logger.info('Sampling context edges from the graph...')
        
        edges = []
        for w in pbar(walks, total=self.G.num_nodes):
            for i in range(len(w)):
                j1 = max(0, i - self.ws)
                j2 = min(len(w), i + self.ws + 1)
                for j in range(j1, j2):
                    # i: center node, j: context node
                    if i == j: continue
                    edges.append([w[i], w[j]])
                    
        logger.info('Sampled %d context edges.' % len(edges))
        return np.array(edges)

    def train(self, epochs=100):
        for epoch in range(epochs):
            logger.info('Epoch: %d' % epoch)
            logger.info('Learning rate = %.2e' % self.lr)
            
            start_time = time()

            walks = self.sample_walks()
            context = self.context_graph(walks)
            
            self.backprop_loss(context)
            self.anneal()
            
            end_time = time()
            logger.info('Epoch time cost: %d' % (1000 * (end_time - start_time)))
            
    def log_softmax(self, u, v):
        """log p(u|v) where p is the hierarchical softmax function"""
        lp = torch.tensor(0.)  # log probability
        n = u
        while True:
            p = self.T.parent[n]
            if p < 0: break
            s = 1 - self.T.code[n] * 2
            x = torch.dot(self.Z1[v], self.Z2[p-self.N])
            lp += torch.sigmoid(s * x).log()
            n = p
        return lp

    def backprop_loss(self, context):
        logger.info('Computing and back propagating loss...')
        
        total_loss = 0
        num_batches = len(context) // self.bs
        
        for i in pbar(range(num_batches)):
            batch_loss = 0
            
            for u, v in context[i*self.bs : (i+1)*self.bs]:
                loss = -self.log_softmax(v, u)
                batch_loss += loss
                total_loss += loss
                
            batch_loss.backward()
            self.sgd()
            
        logger.info('Loss = %.3e' % total_loss)
        
    def sgd(self):
        """Stochastic gradient descent."""
        with torch.no_grad():
            self.Z1 -= self.Z1.grad * self.lr
            self.Z2 -= self.Z2.grad * self.lr
            self.Z1.grad.zero_()
            self.Z2.grad.zero_()
            
    def anneal(self):
        """Learning rate simulated annealing."""
        self.lr *= self.tau

    def similarity(self, u, v):
        with torch.no_grad():
            Zu = self.Z1[u]
            Zv = self.Z1[v]
            return (Zu@Zv) / np.sqrt((Zu@Zu) * (Zv@Zv))


if __name__ == "__main__":
    g = read_graph('datasets/sample_data.txt')
    dw = DeepWalk(g)
    dw.train(epochs=10)

    # print('Similarities:')
    # for i in random.sample(g.nodes, 5):
    #     for j in random.sample(g.nodes, 5):
    #         print('%d -- %d: %.3f' % (i, j, dw.similarity(i, j)))
