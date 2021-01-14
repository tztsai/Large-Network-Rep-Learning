import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from random import sample
from NetMF import NetMF
from sklearn.metrics import roc_auc_score
from utils.txtGraphReader import txtGreader 
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

EMBEDDING_PATH = "./test/NetMF_embedding_decoded.txt"
GRAPH_PATH = "./test/NetMF_graph.txt"
LABEL_PATH = "./test/NetMF_label.txt"


class LinkPrediction():
    def __init__(self):
        self.embeddings = None
        self.graph = None
        self.labels = None
        self.data_loader = None

    def read_file(self, embedding_path, graph_path, label_path):
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

        # read labels
        self.labels = self.process_labels(label_path)
        self.labels = np.array(sorted(self.labels, key=(lambda x:x[0]), reverse=False))

        # return value
        return self.embeddings, self.graph, self.labels

    def process_labels(self, label_path):
        all_labels = set()
        # read label
        y = []
        name = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                values = [int(x.strip()) for x in line.split()]
                y.append(values[1:])
                name.append(np.array([values[0]]))
                for value in values[1:]:
                    all_labels.add(value)
            y = np.array(y)
            name = np.array(name)

        # tranform to boolean matrix
        boolean_matrix = np.zeros((len(y),len(all_labels)))
        for i in range(len(y)):
            for value in y[i]:
                boolean_matrix[i][value] = 1

        # assemble
        res = np.hstack((name, boolean_matrix))
        return res

    def preprocess_graph(self, graph):
        # graph partition
        frac = FRACTION_REMOVE_EDGE 
        remove_size = int(frac * len(graph.edges))
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
        negative_edges = []
        for i in range(remove_size):
            # sample negative edges according to weight
            flag = True
            while flag:
                index1 = self.data_loader.node_sampling.draw()
                index2 = self.data_loader.node_sampling.draw()
                flag = graph.has_edge(self.data_loader.nodedict[index1], self.data_loader.nodedict[index2])
            negative_edges.append((self.data_loader.nodedict[index1], self.data_loader.nodedict[index2]))
        # assemble test split
        testsplit = []
        for item in removed_edges:
            testsplit.append([item,1])
        for item in negative_edges:
            testsplit.append([item,-1])
        
        return graph, testsplit

    def cal_distance(self, graph):
        print(graph.edges(data=True))
        print(graph.nodes)
        N = len(graph.nodes)
        dist_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                n1 = list(graph.nodes)[i]
                n2 = list(graph.nodes)[j]
                if DISTANCE_TYPE == 0:
                    dist_mat[i][j] = self.common_neighbors(graph, n1, n2)
                elif DISTANCE_TYPE == 1:
                    dist_mat[i][j] = self.jaccards_coefficient(graph, n1, n2)
                elif DISTANCE_TYPE == 2:
                    dist_mat[i][j] = self.adamic_adar_score(graph, n1, n2)
                else:
                    dist_mat[i][j] = self.preferential_attachment(graph, n1, n2)

        # to find the index of max value in 2*2 mat
        print(np.where(dist_mat==np.max(dist_mat)))

        # determine how many edges to predict

        # while loop

        # argmax to find edge in the matrix, save edge and rank value

        # set max to -inf




    def get_ROC_AUC_score(self, y_true, y_score):
        return roc_auc_score(y_true, y_score)

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


if __name__ == "__main__":
    lp = LinkPrediction()
    e, g, l = lp.read_file(EMBEDDING_PATH, GRAPH_PATH, LABEL_PATH)
    g, testsplit = lp.preprocess_graph(g)
    dist = lp.cal_distance(g)
    # lp.get_ROC_AUC_score()
