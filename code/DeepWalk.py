import random
import numpy as np
from utils.graph import Graph, read_graph
import config


class DeepWalk:
    ws = 3      # window size
    wl = 5      # walk length
    epochs = 10
    
    def __init__(self, graph: Graph):
        self.G = graph
        self.N = graph.num_nodes
        self.D = config.D
        self.Z = np.random.rand((self.N, self.D))
        
    def sample_walks(self):
        """ Generate a sample of random walks for each node. """
        def walk(v):
            seq = [v]
            for _ in range(self.wl):
                v = self.G.rand_neighbor(v)
                seq.append(v)
            return seq

        return [walk(v) for v in self.G.nodes]

    def deep_walk(self):
        for _ in range(self.epochs):
            walks = self.sample_walks()
            random.shuffle(walks)
            self.skip_gram(walks)

    def skip_gram(self, walks):
        for walk in walks:
            for i, v in enumerate(walk):
                j1, j2 = i-self.ws, i+self.ws+1
                if j1 < 0: j1 = 0
                if j2 > len(walk): j2 = len(walk)
                for j in range(j1, j2):
                    u = walk[j]
                    # TODO: implement the SGD

if __name__ == "__main__":
    g = read_graph('small.txt')
    dw = DeepWalk(g)
    print('sample walks:', dw.sample_walks())
