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
from utils.funcs import pbar, norm, cos_similarity, log_sigmoid, init_param
from DeepWalk import RandomWalk
import config

logger = logging.getLogger('GraphSage')
device = config.DEVICE


class NegSampling:
    """Negative sampling to compute graph embedding loss."""
    wl = 5      # walk length
    ws = 2      # window size
    K = config.NUM_NEG_SAMPLES

    def __init__(self, graph: Graph):
        self.G = graph
        self.S = RandomWalk(graph, self.wl, self.ws)

    def log_prob(self, u, v, Z):
        pos_lp = torch.sigmoid(Z[u] @ Z[v]).log()
        neg_lp = 0.
        for _ in range(self.K):
            vn = self.G.noise_sample()
            neg_lp += torch.sigmoid(-Z[u] @ Z[vn]).log()
        return pos_lp + neg_lp

    def __call__(self, batch, embedding):
        sample = self.S.sample(batch)
        return -sum(self.log_prob(u, v, embedding)
                    for u, v in sample)


class GraphSage(nn.Module):
    bs = 32                 # batch size
    lr = config.ALPHA       # learning rate
    K = 2                   # maximum search depth
    S = [25, 10]            # neighborhood sample size for each search depth

    def __init__(self, graph: Graph, emb_dim=config.D, model_file=None, device=device):
        """
        GraphSAGE neural network.

        Args:
            graph: graph to learn
            emb_dim: dimension of embedding space
            features:
        """
        super().__init__()

        self.N, self.D = N, D = graph.num_nodes, emb_dim
        self.G = graph

        # weight matrices
        self.W = [init_param(D, 2*D) for _ in range(self.K)]

        # embedding matrix
        self.Z = np.random.rand(self.N, self.D)

        self.to(device)

        self.loss = NegSampling(self.G)
        self.opt = optim.Adam(self.W, lr=self.lr)

        if model_file:
            try: self.load(model_file)
            except FileNotFoundError: pass
        
    def forward(self, batch):
        """
        Forward propagation of the neural network.
        
        Args:
            batch: the nodes in the mini-batch
        """

        # record the result so that it returns the same
        # result during a single call of "forward" method
        _ns = {}  # neighborhood samples
        def sample_neighbors(v, k):
            if (v, k) in _ns:
                return _ns[v, k]
            s = self.S[k]
            sample = graph.sample_neighbors(v, s)
            _ns[v, k] = sample
            return sample

        # TODO: deal with features

        B = [set(batch) for _ in range(self.K)]
        for k in range(self.K-1, 0, -1):
            B[k-1].update(B[k])
            for v in B[k]:
                B[k-1].update(sample_neighbors(v, k))
        
        for k in range(self.K):
            newZ = torch.empty(*self.Z.shape)
            s = self.S[k]
            for v in batch:
                neighbors = sample_neighbors(v, k)
                zn = self.aggregate([self.Z[u] for u in neighbors])
                z = self.sigma(self.W[k] @ torch.cat((self.Z[v], zn)))
                newZ[v] = z / norm(z)

        self.Z[batch] = newZ.data[batch]
        return self.loss(batch, newZ)

    def aggregate(self, neighbors):
        return mean_aggregator(neighbors)

    def sigma(self, x):
        return torch.sigmoid(x)

    def fit(self, epochs=100):
        nodes = torch.tensor(self.G.nodes, device=device)
        batches = DataLoader(nodes, batch_size=self.bs)

        for epoch in range(epochs):
            logger.info('\nEpoch: %d' % epoch)
            start_time = time()
            epoch_loss = 0
            
            for batch in pbar(batches):
                loss = self.forward(batch)
                loss.backward()
                epoch_loss += loss
                self.opt.step()
                self.opt.zero_grad()

            logger.info('Loss = %.3e' % epoch_loss)
            end_time = time()
            logger.info('Time cost: %dms' %
                        1000 * (end_time - start_time))

    def save(self, path):
        logger.info(f'Saving model to {path}')
        torch.save(self.state_dict(), path)
            
    def load(self, path):
        logger.info(f'Loading model from {path}')
        state = torch.load(path)
        self.load_state_dict(state)

    def embedding(self):
        """The graph embedding matrix."""
        return self.Z
        
    def save_embedding(self, path):
        logger.info(f'Saving embedding array to {path}')
        emb = self.embedding()
        np.savetxt(path, emb, header=str(emb.shape))

    def similarity(self, u, v):
        Z = self.embedding()
        return cos_similarity(Z[u], Z[v])


# GraphSage aggregators

def mean_aggregator(h):
    return sum(h) / len(h)

def LSTM_aggregator(h):
    pass

def pool_aggregator(h):
    pass


if __name__ == "__main__":
    try:
        data_path = sys.argv[1]
    except IndexError:
        data_path = 'datasets/small.txt'

    dataset = os.path.basename(data_path).split('.')[0]
    print('Dataset:', dataset, end='\n\n')

    model_file = f'models/{dataset}_deepwalk.pt'
    array_file = f'models/{dataset}_deepwalk.txt'

    print('Using device:', device)

    graph = read_graph(data_path)
    # model = GraphSage(graph, model_file=model_file)
    model = GraphSage(graph)

    try:
        model.fit()
    except KeyboardInterrupt:
        print('Training stopped.')
    finally:
        model.save(model_file)
        model.save_embedding(array_file)
