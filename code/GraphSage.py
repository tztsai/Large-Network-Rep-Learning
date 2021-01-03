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
from utils.funcs import norm, cos_similarity
import config

logger = logging.getLogger('GraphSage')


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
        
        if model_file:
            try: self.load(model_file)
            except FileNotFoundError: pass
        
    def forward(self, graph: Graph, features=None):
        """
        Forward propagation of the neural network.
        
        Args:
            graph: the graph to learn
            features (optional): feature of each node
        """
        if features:
            Z = features
        else:
            Z = np.ones(self.N, self.D)

        for k in range(self.K):
            # newZ = np.empty(Z.shape)
            s = self.S[k]
            for v in graph.nodes:
                neighbors = graph.sample_neighbors(v, s)
                zn = self.aggregate([Z[u] for u in neighbors])
                z = self.sigma(self.W[k](np.concatenate((Z[v], zn))))
                # newZ[v] = z / norm(z)
                Z[v] = z / norm(z)
            # Z = newZ

        return Z

    def fit(self, graph, features=None, epochs=100):
        for epoch in range(epochs):
            self.forward(graph, features)

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
