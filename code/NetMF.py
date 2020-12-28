import numpy as np
from utils.graph import Graph, read_graph

# Parameters for NetMF
PARAMETER_T = 1 # window size, 10 for option
PARAMETER_b = 20 # number of negative samples
PARAMETER_h = 256 # number of eigenpairs (rank), 16384 for Flickr
PARAMETER_ns = 1 # negative sample value, 5 for option

# Parameters for DeepWalk (comparison)
PARAMETER_wl = 40 # walk length 40
PARAMETER_nw = 80 # the number of walks starting from each vertex
PARAMETER_ed = 128 # embedding dimension


class NetMF():
    def __init__(self, graph: Graph):
        self.G = graph

    def get_adjacency_matrix(self, G):
        A = np.zeros((G.num_nodes, G.num_nodes))
        # A = A - 1
        for key in G.neighbors:
            for neighbor_key in G.neighbors[key]:
                A[key][neighbor_key] = G.neighbors[key][neighbor_key]
        return A

    def get_degree_matrix(self, G):
        D = np.zeros(G.num_nodes)
        for key in G.neighbors:
            D[key] = len(G.neighbors[key])
        D = np.diag(D)
        return D

    def NetMF_small_T(self):
        # compute adjacency matrix A
        A = self.get_adjacency_matrix(self.G)

        # compute degree matrix D
        D = self.get_degree_matrix(self.G)

        # compute P

        # step 1

        # step 2

        # step 3

        # step 4

    def NetMF_large_T(self):
        pass

if __name__ == "__main__":
    #g = read_graph('small.txt')
    g = read_graph('small_undirected_weighted.txt')
    nmf = NetMF(g)
    nmf.NetMF_small_T()
    nmf.NetMF_large_T()

