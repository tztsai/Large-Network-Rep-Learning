from scipy import linalg, sparse
import numpy as np
from utils.graph import Graph, read_graph

# Global
PRINT_RESULT = True
# SAVE_PATH = "./test/blogcatalog_NetMF_embedding.txt"
# SAVE_PATH_DECODED = "./test/blogcatalog_NetMF_embedding_decoded.txt"

SAVE_PATH = "./test/lesmis_NetMF.txt"
SAVE_PATH_DECODED = "./test/lesmis_NetMF_decoded.txt"


# Parameters for NetMF
PARAMETER_T = 10 # window size, 1, 10 for option
PARAMETER_b = 1 # number of negative samples(ns), 1, 5 for option
PARAMETER_d = 128 # dimension of embedding space
PARAMETER_h = 256 # number of eigenpairs (rank), 16384 for Flickr, 256 for others


class NetMF():
    def __init__(self, graph: Graph):
        self.G = graph
        self.T = PARAMETER_T
        self.b = PARAMETER_b
        self.d = PARAMETER_d
        self.h = PARAMETER_h

    def get_adjacency_matrix(self, G):
        A = np.zeros((G.num_nodes, G.num_nodes))
        # A = A - 1
        for key in G.weights:
            for neighbor_key in G.weights[key]:
                A[key][neighbor_key] = G.weights[key][neighbor_key]
        return A

    def get_degree_matrix(self, G):
        D = np.zeros(G.num_nodes)
        for key in G.neighbors:
            D[key] = len(G.neighbors[key])
        D = np.diag(D)
        return D

    def NetMF_small_T(self, G):
        # compute adjacency matrix A
        A = self.get_adjacency_matrix(self.G)

        # compute degree matrix D
        D = self.get_degree_matrix(self.G)

        # step 1
        P_0 = np.dot(np.linalg.inv(D), A)
        P = []
        for i in range(1, self.T+1):
            P.append(np.linalg.matrix_power(P_0, i))
        P = np.array(P)

        # step 2
        vol_G = np.sum(A)
        sum_P = np.zeros((G.num_nodes, G.num_nodes))
        for mat in P:
            sum_P += mat
        const = vol_G / (self.b * self.T)
        M = const * np.dot(sum_P, np.linalg.inv(D))

        # step 3
        M_prime = np.maximum(M, 1)

        # step 4
        log_M_prime = np.log(M_prime)
        U, Sigma, V_T = np.linalg.svd(log_M_prime)
        U_d = U[:, :self.d]
        Sigma_d = np.diag(Sigma[:self.d])
            
        # step 5
        return np.dot(U_d, np.sqrt(Sigma_d))

    def NetMF_large_T(self, G):
        # compute adjacency matrix A
        A = self.get_adjacency_matrix(self.G)

        # compute degree matrix D
        D = self.get_degree_matrix(self.G)

        # step 1
        D_prime = np.linalg.inv(np.sqrt(D))
        Ed =  np.linalg.multi_dot([D_prime, A, D_prime])
        Lambda, U = linalg.eig(Ed, left=True, right=False)
        U_h = U[:, :self.h]
        Lambda_h = np.diag(Lambda[:self.h])
        Lambda_h = Lambda_h.astype(np.float64)

        # step 2
        vol_G = np.sum(A)
        const = vol_G / self.b
        S = []
        for i in range(1, self.T+1):
            S.append(np.linalg.matrix_power(Lambda_h, i))
        S = np.array(S)
        sum_S = np.zeros((Lambda_h.shape[0], Lambda_h.shape[1]))
        for mat in S:
            sum_S += mat
        sum_S = sum_S / self.T
        M_hat = const * np.linalg.multi_dot(
                [D_prime, U_h, sum_S, np.transpose(U_h), D_prime])
 
        # step 3
        M_hat_prime = np.maximum(M_hat, 1)

        # step 4
        log_M_hat_prime = np.log(M_hat_prime)
        U, Sigma, V_T = np.linalg.svd(log_M_hat_prime)
        U_d = U[:, :self.d]
        Sigma_d = np.diag(Sigma[:self.d])
            
        # step 5
        return np.dot(U_d, np.sqrt(Sigma_d)).astype(np.float64)

    def save_embedding(self, embedding, path):
        name = np.zeros((self.G.num_nodes, 1))
        for i in range(self.G.num_nodes):
            name[i][0] = i
        res = np.hstack((name, embedding))
        np.savetxt(path, res)
    
    def decode_embeddings(self, graph: Graph, load_path, save_path):
        with open(load_path) as f:
            lines = [line[:-1] for line in f.readlines()]
    
        new_embed = []
        for line in lines:
            no, *embeddings = line.split(' ')
            new_embed.append([graph.decode[int(eval(no))], *embeddings])

        with open(save_path, 'w') as f:
            for line in new_embed:
                for cnt, ele in enumerate(line):
                    if cnt == len(line) - 1:
                        f.write(str(ele) + '\n')
                    else:
                        f.write(str(ele) + ' ')


if __name__ == "__main__":
    #g = read_graph('./datasets/small.txt')
    #g = read_graph('./datasets/small_undirected_weighted.txt')
    #g = read_graph('./test/blogcatalogedge.txt')
    #g = read_graph('./test/NetMF_graph.txt')
    g = read_graph('./datasets/lesmis/lesmis.mtx')
    #g = read_graph('./datasets/BlogCatalog.txt')
    #g = read_graph('./datasets/com-youtube.ungraph.txt')
    
    nmf = NetMF(g)
    # res_small = nmf.NetMF_small_T(g)
    res_large = nmf.NetMF_large_T(g)
    # if PRINT_RESULT:
        # print(res_small)
        # print("")
        # print(res_large)
    # nmf.save_embedding(res_small, SAVE_PATH)
    nmf.save_embedding(res_large, SAVE_PATH)
    nmf.decode_embeddings(g, SAVE_PATH, SAVE_PATH_DECODED)
