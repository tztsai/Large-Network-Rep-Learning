import numpy as np
import random
import math
from utils.graph import Graph, read_graph, alias_setup, alias_draw
import numpy.random as npr


# Parameters
PARAMETER_w = 1 # window size

NUM_WALKS = 2
WALK_LENGTH = 10
P = 1
Q = 1
WEIGHTED=True


class Node2vec():
    def __init__(self, G:Graph, p, q, weighted=False):
        self.G = G
        self.p = p # return parameter
        self.q = q # in-out parameter
        self.num_nodes = G.num_nodes
        self.num_edges = G.num_edges
        self.weighted = weighted


    def compute_transfer_prob(self):
        # transfer_prob_nodes[t][v]
        self.transfer_prob_nodes = [] # transfer probability given a previous node
        for node in self.G.neighbors.keys():
            m = [ ]
            if self.weighted:
                for neighbor in self.G[node].keys():
                    k = [ ]

                    for next_neighbor, weight in self.G[neighbor].items():
                        # compute probability for each edge 
                        if next_neighbor == node:
                            k.append(weight * (1/self.p))
                        elif next_neighbor in self.G[node].keys():
                            k.append(weight * 1)
                        else:
                            k.append(weight * (1/self.q)) # transfer prob for 2 hop nodes
                
                    k = [i/sum(k) for i in k] # normalize  distrition k given node v(neighbor) and previous node(t)
                    m.append(alias_setup(k))
                self.transfer_prob_nodes.append(m)

            else:
                pass

    def learn_features(self, num_walks, walk_length):
        walks = []
        nodes = list(self.G.neighbors.keys())
        for iter in range(num_walks):
            print('iteration ', iter+1, '/', num_walks, '...')
            # random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vec_walk(node, walk_length)
                walks.append(walk)
        return walks


    def node2vec_walk(self, start_node, walk_length):
        walk = [start_node]
        current_node = start_node
        while len(walk) < walk_length:
            current_node = walk[-1]
            if len(walk) == 1:
                next_node_index = random.randrange(0, len(self.G[start_node]))
            else:
                prev_node = walk[-2]
                print('pre, cur : ', prev_node, current_node)
                next_node_index = alias_draw(*self.transfer_prob_nodes[prev_node][current_node])
            
            next_node = list(self.G[start_node].keys())[next_node_index]
            walk.append(next_node)
        
       
        return walk


# random walk return a list whose length is (num of nodes x num of walks)
# each element in the list is a list whose length is (walk length)
# i.e. for every node we walk num of walk times with a walk length 


if __name__ == "__main__":

    graph = read_graph('datasets/lesmis/lesmis.mtx', directed=False)
    print(graph[1])
    n2v = Node2vec(graph, P, Q, WEIGHTED)
    n2v.compute_transfer_prob()
    # print(len(n2v.transfer_prob_nodes))
    print(len(n2v.transfer_prob_nodes[1]))
    # walks = n2v.learn_features(NUM_WALKS, WALK_LENGTH)
    # print(walks)
