import random
import logging
import sys
import os
import numpy as np
from gensim.models import Word2Vec
from time import time
from utils.graph import Graph, read_graph
from utils.funcs import *
import config

logger = logging.getLogger('DeepWalk')
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)


class RandomWalk:
    """Randomly walk in the graph to generate samples."""
    
    def __init__(self, graph: Graph, walklen=None):
        self.G = graph
        self.N = graph.num_nodes

        if walklen is None:
            self.wl = config.WALK_LEN
        else:
            self.wl = walklen

    def walk(self, nodes=None):
        """Generate a random walk for each node."""
        def walk(v):
            seq = [str(v)]
            for _ in range(self.wl):
                v = self.G.sample_neighbors(v)
                seq.append(str(v))
            return seq

        if nodes is None:
            nodes = self.G.nodes

        return map(walk, nodes)


class DeepWalk:
    ws = config.WINDOW_SIZE

    def __init__(self, graph: Graph, emb_dim=config.D):
        self.N, self.D = N, D = graph.num_nodes, emb_dim
        self.G = graph
        self.Z = None  # embedding matrix
        self.randwalk = RandomWalk(graph)

    @timer
    def fit(self, iters=10):
        logger.info('Scanning the graph for %d iterations', iters)
        
        walks = []
        logger.info('Generating walks...')
        for i in pbar(range(iters), log_level=logging.INFO):
            walks.extend(self.randwalk.walk())
            
        logger.info('Fitting embedding...')
        model = Word2Vec(walks, size=self.D, window=self.ws)
        self.Z = np.array([
            model.wv[str(i)] for i in range(self.N)
        ])

    def save_embedding(self, path):
        logger.info(f'Saving embedding array to {path}')
        write_embedding(path, self.G, self.Z)

    def similarity(self, u, v):
        Z = self.embedding()
        return cos_similarity(Z[u], Z[v])


if __name__ == "__main__":
    try:
        data_path = sys.argv[1]
    except IndexError:
        data_path = 'datasets/lesmis/lesmis.mtx'
        
    try:
        iterations = int(sys.argv[2])
    except IndexError:
        iterations = 10

    dataset = data_path.split('/')[-2]
    print('Dataset:', dataset, end='\n\n')

    model_file = f'models/{dataset}_deepwalk.pt'
    emb_file = f'results/{dataset}/{dataset}_deepwalk.txt'

    print('Using device:', device)

    graph = read_graph(data_path)
    model = DeepWalk(graph)

    try:
        model.fit(iterations)
    except KeyboardInterrupt:
        print('Training stopped.')
    finally:
        model.save_embedding(emb_file)
