import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from NetMF import NetMF
from sklearn.metrics import roc_auc_score
from utils.txtGraphReader import txtGreader 
from utils.process_labels import *
#from gae.preprocessing import mask_test_edges

# Global                                                                          
EMBEDDING_PATH = "./test/blogcatalog_NetMF_embedding_decoded.txt"
# EMBEDDING_PATH = "./test/node2vec_blogcatalog.embed"
GRAPH_PATH = "./test/blogcatalogedge.txt"
LABEL_PATH = "./test/blogcataloglabel.txt"

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
        self.labels = process_labels(label_path)
        self.labels = np.array(sorted(self.labels, key=(lambda x:x[0]), reverse=False))

        # return value
        return self.embeddings, self.graph, self.labels

    def preprocess_graph(self):
        sm = nx.to_scipy_sparse_matrix(self.graph)

        print(sm)


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
    lp.read_file(EMBEDDING_PATH, GRAPH_PATH, LABEL_PATH)
    # lp.preprocess_graph()
    # lp.get_ROC_AUC_score()
