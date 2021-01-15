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
from utils.funcs import *
import config

logger = logging.getLogger('DeepWalk')
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
# torch.autograd.set_detect_anomaly(True)


class RandomWalk:
    """Randomly walk in the graph to draw samples."""
    wl = 6                  # walk length
    ws = 2                  # window size
    
    def __init__(self, graph: Graph, walklen=None, windowsize=None):
        self.G = graph
        self.N = graph.num_nodes
        
        if walklen is None:
            self.wl = RandomWalk.wl
        else:
            self.wl = walklen

        if windowsize is None:
            self.ws = RandomWalk.ws
        else:
            self.ws = windowsize

    def walk(self, nodes=None):
        """Generate a random walk for each node."""
        def walk(v):
            seq = [v]
            for _ in range(self.wl):
                v = self.G.sample_neighbors(v)
                seq.append(v)
            return seq

        if nodes is None:
            nodes = self.G.nodes

        return map(walk, nodes)
    
    def sample(self, nodes=None):
        """Sample context edges."""
        logger.debug('Sampling context edges from the graph...')

        if nodes is None:
            nodes = self.G.nodes

        edges = []
        walks = self.walk(nodes)
        for w in pbar(walks, total=len(nodes)):
            for i in range(self.ws, self.wl - self.ws):
                j1, j2 = i - self.ws, i + self.ws + 1
                for j in range(j1, j2):
                    # i: center node, j: context node
                    if i == j: continue
                    edges.append([w[i], w[j]])

        logger.debug('Sampled %d context edges.' % len(edges))
        return np.array(edges)
    
    
class HLogSoftMax:
    """Hierarchical log softmax loss of the graph embedding."""
    
    def __init__(self, graph, leaf_emb, inner_emb):
        self.N = graph.num_nodes
        self.Z1 = leaf_emb
        self.Z2 = inner_emb

        # build a Huffman binary tree from graph nodes
        node_weights = [graph.weight(n) for n in graph.nodes]
        self.T = HuffmanTree(node_weights)
        
    def log_prob(self, context):
        """log p(u|v) where p is the hierarchical softmax function"""
        u, v = context
        T, Z1, Z2 = self.T, self.Z1, self.Z2
        lp = 0.
        n = u
        while True:
            p = T.parent[n]
            if p < 0: break
            s = 1 - T.code[n] * 2  # left child: 1, right child: -1
            x = torch.dot(Z1[v], Z2[p-self.N])
            lp += torch.sigmoid(s*x).log()
            n = p
        return lp
    
    def __call__(self, sample):
        return -sum(map(self.log_prob, sample))


class DeepWalk(nn.Module):
    bs = 128                # batch size
    lr = config.ALPHA       # learning rate
    
    def __init__(self, graph: Graph, emb_dim=config.D, model_file=None, device=device):
        super().__init__()

        self.N, self.D = N, D = graph.num_nodes, emb_dim
        self.G = graph
        
        # embedding tensor
        self.Z1 = init_param(N, D)
        self.Z2 = init_param(2*N-1, D)
        # 2*N-1 nodes in the binary tree, the first N are graph nodes
        self.to(device)  # use CPU or GPU device

        self.loss = HLogSoftMax(graph, self.Z1, self.Z2)
        self.opt = optim.Adam(self.parameters(), lr=self.lr)
        self.sampler = RandomWalk(graph)
        
        if model_file:
            try: self.load(model_file)
            except FileNotFoundError:
                logger.warning('Model file "%s" not found. A new model is initialized.')

    @timer
    def fit(self, epochs=10, epoch_iters=10):
        for epoch in range(epochs):
            print()
            logger.info('Epoch: %d' % epoch)
            start_time = time()

            sample = self.sampler.sample()
            batches = DataLoader(sample, batch_size=self.bs)

            for i in range(epoch_iters):
                print()
                logger.info(f'\tIteration: {i}')
                total_loss = 0

                for batch in pbar(batches):
                    loss = self.loss(batch)
                    loss.backward()
                    total_loss += loss
                    self.opt.step()
                    self.opt.zero_grad()

                logger.info('\tLoss = %.3e' % total_loss)

            end_time = time()
            logger.info('Time cost: %dms' % int(1000*(end_time - start_time)))
        
    def save(self, path):
        logger.info(f'Saving model to {path}')
        torch.save(self.state_dict(), path)
            
    def load(self, path):
        logger.info(f'Loading model from {path}')
        state = torch.load(path)
        self.load_state_dict(state)
        
    def embedding(self):
        """The graph embedding matrix."""
        return self.Z1.data.cpu().numpy()

    def save_embedding(self, path):
        logger.info(f'Saving embedding array to {path}')
        write_embedding(path, self.G, self.embedding())
        
    def similarity(self, u, v):
        Z = self.embedding()
        return cos_similarity(Z[u], Z[v])
    

if __name__ == "__main__":
    try:
        data_path = sys.argv[1]
    except IndexError:
        data_path = 'datasets/example.txt'

    dataset = data_path.split('/')[-2]
    print('Dataset:', dataset, end='\n\n')
    
    model_file = f'models/{dataset}_deepwalk.pt'
    emb_file = f'results/{dataset}/{dataset}_deepwalk.txt'

    print('Using device:', device)

    graph = read_graph(data_path)
    model = DeepWalk(graph, model_file=model_file)
    
    try:
        model.fit()
    except KeyboardInterrupt:
        print('Training stopped.')
    finally:
        model.save(model_file)
        model.save_embedding(emb_file)
