from scipy import linalg, sparse
import numpy as np
from utils.graph import Graph, read_graph

# Global
PRINT_RESULT = True
SAVE_PATH = "./test/blogcatalog_max_vote.txt"

# Parameters for NetMF
PARAMETER_L = 39 # number of labels


class MaxVote():
    def __init__(self, graph: Graph):
        self.G = graph
        self.l = PARAMETER_L

    def max_vote(self, G):
        print(self.G)

    def save_results(self, res, path):
        np.savetxt(path, res)


if __name__ == "__main__":
    #g = read_graph('./datasets/small_undirected_weighted.txt')
    g = read_graph('./test/blogcatalogedge.txt')

    mv = MaxVote(g)
    
