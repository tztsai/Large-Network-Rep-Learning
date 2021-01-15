import sys 
sys.path.append("..") 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import random
# from algorithms import NetMF
from sklearn.metrics import roc_auc_score
from algorithms.utils.txtGraphReader import txtGreader 
from algorithms.utils.funcs import pbar
from sklearn.metrics.pairwise import cosine_similarity
#from gae.preprocessing import mask_test_edges

import pdb
def D(): pdb.set_trace()

# random.seed(1)
# Global                                                                          
FRACTION_REMOVE_EDGE = 0.5
K_TRAINING = 2 # 
K_TEST = 2 # 
DISTANCE_TYPE = 0 # 0 for common neighbors, 1 for Jaccard's coeff, 2 for AA, 3 for PA

# Path settings
# EMBEDDING_PATH = "./test/blogcatalog_NetMF_embedding_decoded.txt"
# EMBEDDING_PATH = "./test/node2vec_blogcatalog.embed"
# GRAPH_PATH = "./test/blogcatalogedge.txt"
# LABEL_PATH = "./test/blogcataloglabel.txt"

EMBEDDING_PATH = sys.argv[1]
GRAPH_PATH = sys.argv[2]


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
        print('preprocess graph...')
        frac = FRACTION_REMOVE_EDGE
        remove_size = int(frac * len(graph.edges))
        print('remove size: ' + str(remove_size))
        negative_edges = []
        print('negative sampling...')
        for i in range(remove_size):
            # sample negative edges according to weight
            flag = True
            while flag:
                index1 = self.data_loader.node_sampling.draw()
                index2 = self.data_loader.node_sampling.draw()
                if not index1 == index2:   
                    flag = graph.has_edge(self.data_loader.nodedict[index1], self.data_loader.nodedict[index2])
            negative_edges.append((self.data_loader.nodedict[index1], self.data_loader.nodedict[index2]))     

        removed_edges = set()
        all_edges = list(graph.edges)
        print('remove edges...')
        for _ in pbar(range(remove_size)):
            # remove edge
            while True:
                # import pdb; pdb.set_trace()
                u, v = e = random.choice(all_edges)
                if e in removed_edges: continue
                # check if isolated
                if graph.degree[u] > 1 and graph.degree[v] > 1:
                    graph.remove_edge(u, v) # update dataloader, data_loader.graph_update(graph)
                    removed_edges.add(e)
                    break
        # balance with negative edges
        # assemble test split
        testsplit = []
        for item in removed_edges:
            testsplit.append([item,1])
        for item in negative_edges:
            testsplit.append([item,-1])
        
        return graph, testsplit       
        
    def common_neighbors(self, graph, n1, n2):
        N1 = set(graph.neighbors(n1))
        N2 = set(graph.neighbors(n2))
        return len(g1 & g2)

    def jaccards_coefficient(self, graph, n1, n2):
        N1 = set(graph.neighbors(n1))
        N2 = set(graph.neighbors(n2))
        return len(N1 & N2) / len(N1 | N2)

    def adamic_adar_score(self, graph, n1, n2):
        N1 = set(graph.neighbors(n1))
        N2 = set(graph.neighbors(n2))
        return sum(1/np.log(len(set(graph.neighbors(n))))
                   for n in N1 & N2)

    def preferential_attachment(self, graph, n1, n2):
        N1 = set(graph.neighbors(n1))
        N2 = set(graph.neighbors(n2))
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

    def evaluate(self, testsplit, embedding_path, simi=0):
        embed_dict = {}
        ranked_dict = {}
        with open(embedding_path) as f:
            lines = [line[:-1] for line in f.readlines()]
            for line in lines:
                no, embedding = line.split(' ', 1)
                embedding = embedding[:-1]
                embedding = [float(i) for i in embedding.split(' ')]
                embed_dict[no] = np.array([embedding])
                     

        for edge, b in pbar(testsplit):
            if simi == 0:
                dist = self.common_neighbors(graph, edge[0], edge[1])[0][0]
            elif simi == 1:
                # print(embed_dict[edge[0]])
                # quit()

                dist = cosine_similarity(embed_dict[edge[0]], embed_dict[edge[1]])[0][0]
                dist = (dist +1) / 2

            elif  simi == 2:
                dist = self.adamic_adar_score(graph, edge[0], edge[1])
            else:
                dist = self.preferential_attachment(graph, edge[0], edge[1])

            ranked_dict[(edge[0], edge[1], b)] = dist
        
        ranked_dict = dict(sorted(ranked_dict.items(), key=lambda rank_dict: rank_dict[1], reverse=True))


        pos_score = []
        neg_socre = []

        for edeg, score in ranked_dict.items():
            if edeg[2] == 1:
                pos_score.append(score)
            else:
                neg_socre.append(score)
        # print(len(neg_socre))
        preds_all = np.hstack((pos_score, neg_socre))
        labels_all = np.hstack((np.ones(len(pos_score)), np.zeros(len(neg_socre))))
        roc_score = roc_auc_score(labels_all, preds_all)        

        return roc_score        
        

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
    
    def prt_graph(self, graph, save_path):
            print('prt graph...')
            self.data_loader.graph_update(graph)
            with open(save_path, 'w') as f:
                for item in self.data_loader.edgewithweight:
                    f.write(item[0] + ' ' + item[1] + ' ' + str(int(item[2]['weight'])) + '\n')
            # print(self.data_loader.edgewithweight)



if __name__ == "__main__":
    lp = LinkPrediction()
    e, g = lp.read_file(EMBEDDING_PATH, GRAPH_PATH)
    g, testsplit = lp.preprocess_graph(g)
    r = lp.evaluate(testsplit, EMBEDDING_PATH,1)
    print(r)
    #r = lp.evaluate(g, testsplit, 1)
    #print(r)
    # lp.get_ROC_AUC_score()
