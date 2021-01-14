import sys 
sys.path.append("..") 

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from random import sample
# from algorithms import NetMF
from sklearn.metrics import roc_auc_score
from algorithms.utils.txtGraphReader import txtGreader 
#from gae.preprocessing import mask_test_edges

# Global                                                                          
FRACTION_REMOVE_EDGE = 0.5
K_TRAINING = 2 # 3
K_TEST = 2 # 3
DISTANCE_TYPE = 0 # 0 for common neighbors, 1 for Jaccard's coeff, 2 for AA, 3 for PA

# Path settings
# EMBEDDING_PATH = "./test/blogcatalog_NetMF_embedding_decoded.txt"
# EMBEDDING_PATH = "./test/node2vec_blogcatalog.embed"
# GRAPH_PATH = "./test/blogcatalogedge.txt"
# LABEL_PATH = "./test/blogcataloglabel.txt"

EMBEDDING_PATH = "../results/lesmis/lesmis_node2vec.txt"
GRAPH_PATH = "../datasets/lesmis/lesmis.mtx"


class LinkPrediction():
    def __init__(self):
        self.embeddings = None
        self.graph = None
        self.labels = None
        self.data_loader = None

    def read_file(self, embedding_path, graph_path):
        # read embedding
        self.embeddings = []
        with open(embedding_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                values = [float(x.strip()) for x in line.split()]
                self.embeddings.append(values)
        self.embeddings = np.array(sorted(self.embeddings, key=(lambda x:x[0]), reverse=False))

        # read graph
        self.data_loader = txtGreader(graph_path, direct=False, weighted=True)
        self.graph = self.data_loader.graph
        self.graph = self.graph.to_undirected()

        # return value
        return self.embeddings, self.graph


    def preprocess_graph(self, graph):
        # graph partition
        frac = FRACTION_REMOVE_EDGE
        remove_size = int(frac * len(graph.edges))
        negative_edges = []
        for i in range(remove_size):
            # sample negative edges according to weight
            flag = True
            while flag:
                index1 = self.data_loader.node_sampling.draw()
                index2 = self.data_loader.node_sampling.draw()
                if index1 == index2:
                    continue
                flag = graph.has_edge(self.data_loader.nodedict[index1], self.data_loader.nodedict[index2])
            negative_edges.append((self.data_loader.nodedict[index1], self.data_loader.nodedict[index2]))     

        removed_edges = []
        for i in range(remove_size):
            removed = False
            # remove edge
            while removed == False:
                cur_e = sample(list(graph.edges), 1)[0]
                # check if isolated
                if graph.degree[cur_e[0]] != 1 and graph.degree[cur_e[1]] != 1:
                    graph.remove_edge(cur_e[0], cur_e[1]) # update dataloader, data_loader.graph_update(graph)
                    removed_edges.append(cur_e)
                    removed = True
        # balance with negative edges
        # assemble test split
        testsplit = []
        for item in removed_edges:
            testsplit.append([item,1])
        for item in negative_edges:
            testsplit.append([item,-1])
        
        return graph, testsplit       
    def common_neighbors(self, graph, n1, n2):
        g1 = set()
        g2 = set()
        for edge in graph.edges:
            # cal neighbors
            if n1 == edge[0]:
                g1.add(edge[1])
            elif n1 == edge[1]:
                g1.add(edge[0])

            # cal neighbors
            if n2 == edge[0]:
                g2.add(edge[1])
            elif n2 == edge[1]:
                g2.add(edge[0])

        return len(g1 & g2)

    def jaccards_coefficient(self, graph, n1, n2):
        g1 = set()
        g2 = set()
        for edge in graph.edges:
            # cal neighbors
            if n1 == edge[0]:
                g1.add(edge[1])
            elif n1 == edge[1]:
                g1.add(edge[0])

            # cal neighbors
            if n2 == edge[0]:
                g2.add(edge[1])
            elif n2 == edge[1]:
                g2.add(edge[0])

        return len(g1 & g2) / len(g1 | g2)

    def adamic_adar_score(self, graph, n1, n2):
        g1 = set()
        g2 = set()
        for edge in graph.edges:
            # cal neighbors
            if n1 == edge[0]:
                g1.add(edge[1])
            elif n1 == edge[1]:
                g1.add(edge[0])

            # cal neighbors
            if n2 == edge[0]:
                g2.add(edge[1])
            elif n2 == edge[1]:
                g2.add(edge[0])
        # construct z
        Z = g1 & g2
        res = 0
        gz = set()
        for z in Z:
            for edge in graph.edges:
                # cal neighbors
                if z == edge[0]:
                    gz.add(edge[1])
                elif z == edge[1]:
                    gz.add(edge[0])
            res += 1/np.log(len(gz))
        return res

    def preferential_attachment(self, graph, n1, n2):
        g1 = set()
        g2 = set()
        for edge in graph.edges:
            # cal neighbors
            if n1 == edge[0]:
                g1.add(edge[1])
            elif n1 == edge[1]:
                g1.add(edge[0])

            # cal neighbors
            if n2 == edge[0]:
                g2.add(edge[1])
            elif n2 == edge[1]:
                g2.add(edge[0])

        return len(g1) * len(g2)

    def cal_distance(self, graph, testsplit, simi=0):
        # with open(embed_path) as f:
        #     lines = [line[:-1] for line in f.readlines()]

        # for line in lines:
        #     no, embeddings = line.split(' ', 1)
        #     no_embed[no] = embeddings

        ranked_dict = {}
        for edge, b in testsplit:
            # ('49', '63') 1
            #  element dict :key list['node1', 'node2', pos/neg]: similarity
            for edge, b in testsplit:
                if simi == 0:
                    dist = self.common_neighbors(graph, edge[0], edge[1])
                elif simi == 1:
                    dist = self.jaccards_coefficient(graph, edge[0], edge[1])
                elif  simi == 2:
                    dist = self.adamic_adar_score(graph, edge[0], edge[1])
                else:
                    dist = self.preferential_attachment(graph, edge[0], edge[1])

                ranked_dict[(edge[0], edge[1], b)] = dist
        ranked_dict = dict(sorted(ranked_dict.items(), key=lambda rank_dict: rank_dict[1], reverse=True))
        
        acc = self.get_acc_posedge(ranked_dict)
        print(acc)


    def get_acc_posedge(self, ranked_dict):
        print(ranked_dict)

        true_edge = 0
        num_pos_edge = int(len(ranked_dict) / 2)
        print(num_pos_edge)
        for cnt, edge in enumerate(ranked_dict.keys()):
            print(cnt, edge)
            if cnt == num_pos_edge:
                print(true_edge, num_pos_edge)
                return true_edge / num_pos_edge
            if edge[2] == 1:
                true_edge += 1
    
    def get_ROC_AUC_score(self, y_true, y_score):
        return roc_auc_score(y_true, y_score)

            
            
        


if __name__ == "__main__":
    lp = LinkPrediction()
    e, g = lp.read_file(EMBEDDING_PATH, GRAPH_PATH)
    g, testsplit = lp.preprocess_graph(g)
    lp.cal_distance(g, testsplit)
    # lp.get_ROC_AUC_score()
