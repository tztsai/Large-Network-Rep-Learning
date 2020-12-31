import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time

from utils.graph import Graph, read_graph
from utils.huffman import HuffmanTree
from utils.funcs import pbar, cos_similarity
import config

logger = logging.getLogger('DeepWalk')

model_file = 'models/sample_data_deepwalk.pt'
array_file = 'models/sample_data_deepwalk.txt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Using device: %s' % device)


class DeepWalk(nn.Module):
    wl = 6                  # walk length
    ws = 2                  # window size
    
    bs = 128                # batch size
    lr = config.ALPHA       # learning rate
    tau = 0.99              # exponential annealing rate
    
    def __init__(self, graph: Graph):
        super().__init__()

        self.G = graph
        self.N = graph.num_nodes
        self.D = config.D

        # build a Huffman binary tree from graph nodes
        node_weights = [sum(graph.neighbors[n].values())
                        for n in graph.nodes]
        self.T = HuffmanTree(node_weights) 
        
        # latent representation of nodes
        self.Z1 = nn.Parameter(torch.rand(self.N, self.D))
        # latent representation of inner nodes of the tree
        self.Z2 = nn.Parameter(torch.rand(self.N-1, self.D))
        
        self.to(device)
        
        # stochastic gradient descent optimizer
        self.opt = optim.SGD(self.parameters(), lr=self.lr)
        
    def walk(self):
        """Generate a random walk for each node."""
        def walk(v):
            seq = [v]
            for _ in range(self.wl):
                v = self.G.rand_neighbor(v)
                seq.append(v)
            return seq

        return map(walk, self.G.nodes)
    
    def sample(self):
        """Sample context edges."""
        logger.info('Sampling context edges from the graph...')

        edges = []
        walks = self.walk()
        for w in pbar(walks, total=self.G.num_nodes):
            for i in range(len(w)):
                j1 = max(0, i - self.ws)
                j2 = min(len(w), i + self.ws + 1)
                for j in range(j1, j2):
                    # i: center node, j: context node
                    if i == j: continue
                    edges.append([w[i], w[j]])

        logger.info('Sampled %d context edges.' % len(edges))
        return torch.tensor(edges, device=device)
    
    def log_softmax(self, u, v):
        """log p(u|v) where p is the hierarchical softmax function"""
        lp = torch.tensor(0.)  # log probability
        n = u
        while True:
            p = self.T.parent[n]
            if p < 0: break
            s = 1 - self.T.code[n] * 2
            x = torch.dot(self.Z1[v], self.Z2[p-self.N])
            lp += torch.sigmoid(s*x).log()
            n = p
        return lp
    
    def loss(self, sample):
        return -sum(self.log_softmax(v, u) for u, v in sample)
    
    def step(self):
        total_loss = 0
        
        sample = self.sample()
        batches = DataLoader(sample, batch_size=self.bs)
        
        for batch in pbar(batches):
            loss = self.loss(batch)
            total_loss += loss

            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        self.anneal()
        logger.info('Loss = %.3e' % total_loss)
        
    def anneal(self):
        """Learning rate simulated annealing."""
        self.lr *= self.tau
        
    def save(self, path):
        torch.save(self.state_dict(), path)
            
    def load(self, path):
        state = torch.load(path)
        self.load_state_dict(state)
        
    def embedding(self):
        return self.Z1.data.cpu().numpy()

    def save_embedding(self, path):
        Z = self.embedding()
        np.savetxt(path, Z, header=str(Z.shape))
        
    def similarity(self, u, v):
        Z = self.embedding()
        return cos_similarity(Z[u], Z[v])
     
        
def train(model: DeepWalk, epochs=100):
    for epoch in range(epochs):
        logger.info('Epoch: %d' % epoch)
        logger.info('Learning rate = %.2e' % model.lr)
        
        start_time = time()
        model.step()
        end_time = time()
        logger.info('Epoch time cost: %dms.\n' %
                    (1000 * (end_time - start_time)))
    

if __name__ == "__main__":
    graph = read_graph('datasets/sample_data.txt')
    model = DeepWalk(graph)
    train(model, epochs=250)
    model.save(model_file)
    model.save_embedding(array_file)
