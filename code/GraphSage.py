import random
import logging
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
from utils.graph import Graph, read_graph
from utils.funcs import pbar, norm, cos_similarity
import config

logger = logging.getLogger('GraphSage')

randseed = 0
random.seed(randseed)


class GraphSage(nn.Module):
    bs = 64                 # batch size
    lr = config.ALPHA       # learning rate
    K = 2                   # maximum search depth
    S = [25, 10]            # neighborhood sample size for each search depth
    sigma = torch.sigmoid   # nonlinearity function
    aggregate = np.mean     # aggregator function
    
    def __init__(self, N, D=config.D, model_file=None):
        """
        GraphSAGE neural network.

        Args:
            N: number of graph nodes
            D: dimension of embedding space
        """
        super().__init__()
        self.N, self.D = N, D

        # weight matrices
        self.W = [nn.Linear(D, 2*D) for _ in range(self.K)]
        
        # embedding matrix
        self.Z = np.random.rand(self.N, self.D)
        
        if model_file:
            try: self.load(model_file)
            except FileNotFoundError: pass
        
    def forward(self, graph: Graph, nodes, features=None):
        """
        Forward propagation of the neural network.
        
        Args:
            graph: the graph to learn
            nodes: the nodes in the minibatch
            features (optional): feature of each node
        """
        
        _ns = {}  # neighborhood samples
        def sample_neighbors(v, k):
            if (v, k) in _ns:
                return _ns[v, k]
            s = self.S[k]
            sample = graph.sample_neighbors(v, s)
            _ns[v, k] = sample
            return sample

        # TODO: deal with features

        B = [set(nodes) for _ in range(self.K)]
        for k in range(self.K-1, 0, -1):
            B[k-1].update(B[k])
            for v in B[k]:
                B[k-1].update(sample_neighbors(v, k))
        
        for k in range(self.K):
            # newZ = np.empty(Z.shape)
            s = self.S[k]
            for v in nodes:
                neighbors = sample_neighbors(v, k)
                zn = self.aggregate([self.Z[u] for u in neighbors])
                z = self.sigma(self.W[k](np.concatenate((self.Z[v], zn))))
                # newZ[v] = z / norm(z)
                self.Z[v] = z / norm(z)
            # Z = newZ

        return Z

    def fit(self, graph, features=None, epochs=100):
        batches = DataLoader(graph.nodes, batch_size=self.bs)

        for epoch in range(epochs):
            print()
            logger.info('Epoch: %d' % epoch)
            start_time = time()
            
            for batch in pbar(batches):
                self.forward(graph, batch, features)


    def save(self, path):
        logger.info(f'Saving model to {path}')
        torch.save(self.state_dict(), path)
            
    def load(self, path):
        logger.info(f'Loading model from {path}')
        state = torch.load(path)
        self.load_state_dict(state)
        
    def save_embedding(self, path):
        logger.info(f'Saving embedding array to {path}')
        emb = self.embedding()
        np.savetxt(path, emb, header=str(emb.shape))

    def similarity(self, u, v):
        Z = self.embedding()
        return cos_similarity(Z[u], Z[v])


def LSTM_aggregator(h):
    pass

def pool_aggregator(h):
    pass
