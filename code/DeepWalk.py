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
from utils.huffman import HuffmanTree
from utils.funcs import pbar, cos_similarity
import config

logger = logging.getLogger('DeepWalk')


class RandomWalk:
    """Randomly walk in the graph to draw samples."""
    wl = 6                  # walk length
    ws = 2                  # window size
    
    def __init__(self, graph):
        self.G = graph
        self.N = graph.num_nodes

    def walk(self):
        """Generate a random walk for each node."""
        def walk(v):
            seq = [v]
            for _ in range(self.wl):
                v = self.G.sample_neighbors(v)
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
    
    
class HLogSoftMax:
    """Hierarchical log softmax loss of the graph embedding."""
    
    def __init__(self, graph):
        self.N = graph.num_nodes

        # build a Huffman binary tree from graph nodes
        node_weights = [graph.weight(n) for n in graph.nodes]
        self.T = HuffmanTree(node_weights)
        
    def log_softmax(self, u, v, Z):
        """log p(u|v) where p is the hierarchical softmax function"""
        lp = torch.tensor(0.)  # log probability
        n = u
        while True:
            p = self.T.parent[n]
            if p < 0: break
            s = 1 - self.T.code[n] * 2
            x = torch.dot(Z[v], Z[p])
            lp += torch.sigmoid(s*x).log()
            n = p
        return lp
    
    def __call__(self, sample, embedding):
        return -sum(self.log_softmax(v, u, embedding)
                    for u, v in sample)
    

class DeepWalk(nn.Module):
    bs = 128                # batch size
    lr = config.ALPHA       # learning rate
    tau = 0.99              # exponential annealing rate
    
    def __init__(self, N, D=config.D, model_file=None):
        super().__init__()
        self.N, self.D = N, D
        
        # embedding tensor
        self.Z = nn.Parameter(torch.rand(2*N-1, D) / np.sqrt(2*N-1))
        # 2*N-1 nodes in the binary tree, the first N are graph nodes
        self.to(device)  # use CPU or GPU device

        self.opt = optim.Adam(self.parameters(), lr=self.lr)
        
        if model_file:
            try: self.load(model_file)
            except FileNotFoundError:
                logger.warning('Model file "%s" not found. A new model is initialized.')
        
    def anneal(self):
        """Learning rate simulated annealing."""
        self.lr *= self.tau
        
    def fit(self, graph, epochs=10, epoch_iters=10):
        sampler = RandomWalk(graph)
        loss_func = HLogSoftMax(graph)

        for epoch in range(epochs):
            print()
            logger.info('Epoch: %d' % epoch)
            start_time = time()

            sample = sampler.sample()
            batches = DataLoader(sample, batch_size=self.bs)

            for i in range(epoch_iters):
                print()
                logger.info(f'\tIteration: {i}')
                total_loss = 0

                for batch in pbar(batches):
                    loss = loss_func(batch, self.Z)
                    total_loss += loss
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()

                # self.anneal()
                logger.info('\tLoss = %.3e' % total_loss)

            end_time = time()
            logger.info('Epoch time cost: %dms.' %
                        (1000 * (end_time - start_time)))
        
    def save(self, path):
        logger.info(f'Saving model to {path}')
        torch.save(self.state_dict(), path)
            
    def load(self, path):
        logger.info(f'Loading model from {path}')
        state = torch.load(path)
        self.load_state_dict(state)
        
    def embedding(self):
        """The graph embedding matrix."""
        return self.Z.data.cpu().numpy()[:self.N]

    def save_embedding(self, path):
        logger.info(f'Saving embedding array to {path}')
        Z = self.embedding()
        np.savetxt(path, Z, header=str(Z.shape))
        
    def similarity(self, u, v):
        Z = self.embedding()
        return cos_similarity(Z[u], Z[v])
    
        
if __name__ == "__main__":
    try:
        data_path = sys.argv[1]
    except:
        data_path = 'datasets/small.txt'

    dataset = os.path.basename(data_path).split('.')[0]
    print('Dataset:', dataset, end='\n\n')
    
    model_file = f'models/{dataset}_deepwalk.pt'
    array_file = f'models/{dataset}_deepwalk.txt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s' % device)

    graph = read_graph(data_path)
    model = DeepWalk(graph.num_nodes, model_file=model_file)
    
    try:
        model.fit(graph)
    except KeyboardInterrupt:
        print('Training stopped.')
    finally:
        model.save(model_file)
        model.save_embedding(array_file)
