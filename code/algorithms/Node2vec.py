import numpy as np
import random
import math
from utils.graph import Graph, read_graph
import numpy.random as npr
from gensim.models import Word2Vec
from config import WINDOW_SIZE, D

# Parameters
PARAMETER_w = 1 # window size

NUM_WALKS = 30
WALK_LENGTH = 80
P = 1
Q = 1


class Node2vec():
    def __init__(self, G:Graph, p, q):
        self.G = G
        self.p = p # return parameter
        self.q = q # in-out parameter
        self.num_nodes = G.num_nodes
        self.num_edges = G.num_edges


    def compute_transfer_prob(self):
        # transfer_prob_nodes[t][v]
        self.transfer_prob_nodes = [] # transfer probability given a previous node
        for node in self.G.nodes:
            m = [ ]
            for neighbor in self.G[node]:
                k = []
                for next_neighbor in self.G[neighbor]:
                    # compute probability for each edge 
                    weight = self.G.weight(next_neighbor)
                    if next_neighbor == node:
                        k.append(weight * (1/self.p))
                    elif next_neighbor in self.G[node]:
                        k.append(weight * 1)
                    else:
                        k.append(weight * (1/self.q)) # transfer prob for 2 hop nodes
            
                k = [i/sum(k) for i in k] # normalize  distrition k given node v(neighbor) and previous node(t)
                m.append(alias_setup(k))

            self.transfer_prob_nodes.append(m)



    def learn_features(self, d, ws, num_walks, walk_length, save_path):
        walks = []
        nodes = self.G.nodes
        for iter in range(num_walks):
            print('iteration ', iter+1, '/', num_walks, '...')
            nodes = random.sample(self.G.nodes, self.G.num_nodes)
            print(nodes)
            quit()
            for node in nodes:
                walk = self.node2vec_walk(node, walk_length)
                walks.append(walk)

        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(walks, size=d, window=ws, min_count=0)
        model.save_word2vec_format(save_path)





    def node2vec_walk(self, start_node, walk_length):
        walk = [start_node]
        current_node = start_node
        while len(walk) < walk_length:
            current_node = walk[-1]
            if len(walk) == 1:
                next_node = random.choice(self.G[start_node])
            else:
                prev_node = walk[-2]
                # print('pre, cur : ', prev_node, current_node)
                current_nodex_index = self.G[prev_node].index(current_node)
                next_node_index = alias_draw(*self.transfer_prob_nodes[prev_node][current_nodex_index])
                next_node = self.G[current_node][next_node_index]
            
            walk.append(next_node)
        
       
        return walk


def alias_setup(probs):
    '''
    this code was taken from this page:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    K       = len(probs)
    q       = np.zeros(K)
    J       = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    this code was taken from this page:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    K  = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand()*K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]

# random walk return a list whose length is (num of nodes x num of walks)
# each element in the list is a list whose length is (walk length)
# i.e. for every node we walk num of walk times with a walk length 
# node classification ---> Micro-F1 and Macro-F1.

def decode_embeddings(graph: Graph, load_path, save_path):
    with open(load_path) as f:
        lines = [line[:-1] for line in f.readlines()]
    
    new_embed = []
    for line in lines:
        no, *embeddings = line.split(' ')
        new_embed.append([graph.decode[int(no)], *embeddings])

    with open(save_path, 'w') as f:
        for line in new_embed:
            for cnt, ele in enumerate(line):
                if cnt == len(line) - 1:
                    f.write(str(ele) + '\n')
                else:
                    f.write(str(ele) + ' ')
        




if __name__ == "__main__":

    graph = read_graph('../datasets/lesmis/lesmis.mtx', directed=False)
    # print(graph.num_nodes)
    # n2v = Node2vec(graph, P, Q)
    # n2v.compute_transfer_prob()

    # n2v.learn_features(128, 10, NUM_WALKS, WALK_LENGTH, '../results/lesmis/lesmis_node2vec.txt')
    decode_embeddings(graph, '../results/lesmis/lesmis_node2vec.txt', '../results/lesmis/lesmis_node2vec.txt')