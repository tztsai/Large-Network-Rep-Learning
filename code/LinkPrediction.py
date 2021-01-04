import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from NetMF import NetMF
from sklearn.metrics import roc_auc_score
from utils.txtGraphReader import txtGreader 
from gae.preprocessing import mask_test_edges

# Global                                                                          
EMBEDDING_PATH = "./results/NetMF_embedding.txt"
GRAPH_PATH = "./results/NetMF_graph.txt"

class LinkPrediction():
    def __init__(self):
        self.embedding = None
        self.graph = None

    def read_file(self, embedding_path, graph_path):
        # read embedding
        self.embedding = []
        with open(embedding_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                values = [float(x.strip()) for x in line.split()]
                self.embedding.append(values)

        # read graph
        self.graph = txtGreader(graph_path, direct=False, weighted=True).graph
        self.graph = self.graph.to_undirected()
        #nx.draw_networkx(self.graph, with_labels=False, node_size=6, node_color='r')
        #plt.show()

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
    lp.read_file(EMBEDDING_PATH, GRAPH_PATH)
    lp.preprocess_graph()
    # lp.get_ROC_AUC_score()
