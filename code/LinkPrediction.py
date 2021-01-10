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
        self.graph = txtGreader(graph_path, direct=False, weighted=True).graph
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
                    graph.remove_edge(cur_e[0], cur_e[1])
                    removed_edges.append(cur_e)
                    removed = True

        # add negative edges, 1 for each removed edge between random vertex pairs
        for i in range(remove_size):
            pass


    def get_ROC_AUC_score(self, y_true, y_score):
        return roc_auc_score(y_true, y_score)

    def common_neighbors(self):
        pass

    def jaccards_coefficient(self):
        pass

    def adamic_adar_score(self):
        pass

    def preferential_attachment(self):
        pass


if __name__ == "__main__":
    lp = LinkPrediction()
    e, g, l = lp.read_file(EMBEDDING_PATH, GRAPH_PATH, LABEL_PATH)
    lp.preprocess_graph(g)
    # lp.get_ROC_AUC_score()
