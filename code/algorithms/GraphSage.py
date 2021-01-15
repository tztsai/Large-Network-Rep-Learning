import random
import logging
import sys
import pdb
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
from utils.graph import Graph, read_graph
from utils.funcs import *
from utils.visualize import plot_loss
from DeepWalk import RandomWalk
import config

logger = logging.getLogger('GraphSage')
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
# torch.autograd.set_detect_anomaly(True)


class NegSampling:
    """Negative sampling to compute graph embedding loss."""
    wl = 5      # walk length
    ws = 2      # window size
    K = config.NUM_NEG_SAMPLES

    def __init__(self, graph: Graph, embedding: torch.Tensor):
        self.G = graph
        self.Z = embedding
        self.S = RandomWalk(graph, self.wl, self.ws)

    def log_prob(self, u, v, z):

        def Z(v):  # node embedding
            return z[v] if v in z else self.Z[v]

        pos_lp = torch.sigmoid(Z(u) @ Z(v)).log()
        neg_lp = torch.tensor(0.)
        for _ in range(self.K):
            vn = self.G.noise_sample()
            neg_lp += torch.sigmoid(-Z(u) @ Z(vn)).log()

        return pos_lp + neg_lp

    def __call__(self, batch, new_emb):
        sample = self.S.sample(batch)
        loss = -sum(self.log_prob(*ctx, new_emb) for ctx in sample)
        return loss


class GraphSage(nn.Module):
    bs = 128                # batch size
    lr = config.ALPHA       # learning rate
    K = 2                   # maximum search depth, also number of layers
    S = [25, 10]            # neighborhood sample size for each search depth
    aggregation = 'pool'    # aggregation of neighbors' information

    def __init__(self, graph: Graph, emb_dim=config.D, model_file=None, device=device):
        """
        GraphSAGE neural network.

        Args:
            graph: graph to learn
            emb_dim: dimension of embedding space
            model_file (str, optional): if specified, try to load model from this file
            device: the device used for computation
        """
        super().__init__()

        self.N, self.D = N, D = graph.num_nodes, emb_dim
        self.G = graph

        # weight matrices
        self.W = [init_param(D, 2*D) for _ in range(self.K)]

        # embedding matrix
        self.Z = torch.rand(self.N, self.D)

        self.to(device)

        self.loss = NegSampling(self.G, self.Z)
        self.opt = optim.Adam(self.W, lr=self.lr)
        
        if self.aggregation == 'mean':
            self.aggregator = MeanAggregator()
        elif self.aggregation == 'pool':
            self.aggregator = PoolAggregator(self.D)
        else:
            raise NotImplementedError

        if model_file:
            try: self.load(model_file)
            except FileNotFoundError: pass

    def forward(self, batch):
        """
        Forward propagation of the neural network.
        
        Args:
            batch: the nodes in the mini-batch
        """

        # store the neighbor samples for duplication
        neighbor_sample = {}

        # collect all needed nodes for each layer
        B = [set(map(int, batch)) for _ in range(self.K+1)]
        for k in range(self.K, 0, -1):
            logger.debug('Sampling neighbors in layer %d', k)
            B[k-1].update(B[k])
            s = self.S[k-1]  # sample size
            for v in B[k]:
                neighbors = neighbor_sample.setdefault(
                    v, self.G.sample_neighbors(v, s))
                B[k-1].update(neighbors)

        # new embeddings
        h = [{} for k in range(self.K+1)]
        h[0].update((v, self.Z[v]) for v in B[0])
        
        # aggregate information layer by layer
        for k in range(1, self.K+1):
            logger.debug('Aggregating context information in layer %d', k)
            for v in B[k]:
                neighbors = neighbor_sample[v]
                neighbors_info = [h[k-1][u] for u in neighbors]
                ha = self.aggregator(neighbors_info)
                hs = torch.sigmoid(self.W[k-1] @ torch.cat([h[k-1][v], ha]))
                h[k][v] = normalize(hs, dim=0)
                
        # gather updated embeddings
        newZ = h[1]
        for k in range(2, self.K+1):
            newZ.update(h[k])

        for v in newZ:
            self.Z[v] = newZ[v].detach()

        logger.debug('Computing Loss...')
        return self.loss(batch, newZ)

    @timer
    def fit(self, epochs=10):
        nodes = torch.tensor(self.G.nodes, device=device)
        batches = DataLoader(nodes, batch_size=self.bs)
        losses = []

        for epoch in range(epochs):
            print()
            logger.info('Epoch: %d' % epoch)
            start_time = time()
            epoch_loss = 0
            progress = 0
            
            for batch in pbar(batches, log_level=logging.INFO):
                loss = self.forward(batch)
                loss.backward()
                epoch_loss += loss
                self.opt.step()
                self.opt.zero_grad()
                
                progress += 100 / len(batches)
                logger.debug('Epoch progress: %d%%\n', int(progress))

            logger.info('Loss = %.3e' % epoch_loss)
            losses.append(epoch_loss)
            end_time = time()
            logger.info('Time cost: %dms' % int(1000*(end_time - start_time)))
            
        return losses

    def save(self, path):
        logger.info(f'Saving model to {path}')
        torch.save(self.state_dict(), path)

    def load(self, path):
        logger.info(f'Loading model from {path}')
        state = torch.load(path)
        self.load_state_dict(state)

    def embedding(self):
        """The graph embedding matrix."""
        return self.Z.data.cpu().numpy()

    def save_embedding(self, path):
        logger.info(f'Saving embedding array to {path}')
        write_embedding(path, self.G, self.embedding())

    def similarity(self, u, v):
        Z = self.embedding()
        return cos_similarity(Z[u], Z[v])


# GraphSage aggregators

class MeanAggregator:
    def __call__(self, neighbors):
        return sum(neighbors) / len(neighbors)


class PoolAggregator:
    def __init__(self, dim):
        self.lin = nn.Linear(dim, dim)
        
    def __call__(self, neighbors):
        h = torch.stack(neighbors)
        return torch.max(torch.sigmoid(self.lin(h)), dim=0)[0] 


if __name__ == "__main__":
    try:
        data_path = sys.argv[1]
    except IndexError:
        data_path = 'datasets/pubmed/pubmed_graph.txt'
        
    try:
        epochs = sys.argv[2]
    except IndexError:
        epochs = 30

    dataset = data_path.split('/')[-2]
    print('Dataset:', dataset, end='\n\n')

    model_file = f'models/{dataset}_graphsage.pt'
    emb_file = f'results/{dataset}/{dataset}_graphsage.txt'

    print('Using device:', device)

    graph = read_graph(data_path)
    model = GraphSage(graph, model_file=model_file)

    try:
        losses = model.fit(epochs=epochs)
        plot_loss(losses)
    except KeyboardInterrupt:
        print('Training stopped.')
    finally:
        model.save(model_file)
        model.save_embedding(emb_file)
