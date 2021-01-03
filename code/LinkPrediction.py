import numpy as np
import networkx as nx
from NetMF import NetMF
from sklearn.metrics import roc_auc_score

# Global                                                                          
PATH = "./results/embeddings/"
FILE_NAME = "NetMF.txt"

class LinkPrediction():
    def __init__(self):
        self.embedding = []

    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                values = [float(x.strip()) for x in line.split()]
                self.embedding.append(values)
            self.embedding = np.array(self.embedding)
    
    def preprocess_graph(self):
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
    lp.read_file(PATH+FILE_NAME)
    lp.preprocess_graph()
    lp.get_ROC_AUC_score()
