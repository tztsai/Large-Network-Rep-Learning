from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from utils.process_labels import *
import numpy as np

# Global                                                                          
SEED = 0
# EMBEDDING_PATH = "./test/blogcatalog_NetMF_embedding.txt"
# EMBEDDING_PATH = "./test/blogcatalogedge_deepwalk.txt"
EMBEDDING_PATH = "./test/node2vec_blogcatalog.embed"
LABEL_PATH = "./test/blogcataloglabel.txt"

class NodeClassification():
    def __init__(self):
        self.X = None
        self.y = None

    def read_file(self, embedding_path, label_path):
        # read embedding
        self.X = []
        with open(embedding_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                values = [float(x.strip()) for x in line.split()]
                self.X.append(values)
            self.X = np.array(self.X)
        # read labels
        self.y = process_labels(label_path)
        return self.X, self.y
    
    def node_classification(self, X, y):
        # one-vs-rest logistic regression
        clf = OneVsRestClassifier(LogisticRegression(random_state=SEED))
        # report Micro-F1 and Macro-F1 scores
        ma_scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
        mi_scores = cross_val_score(clf, X, y, cv=5, scoring='f1_micro')
        return np.mean(ma_scores), np.mean(mi_scores)


if __name__ == "__main__":
    nc = NodeClassification()
    X, y = nc.read_file(EMBEDDING_PATH, LABEL_PATH)
    ma_score, mi_score = nc.node_classification(X, y)
    print("Accurancy for node classification: ")
    print("Micro-F1 score: ", mi_score)
    print("Macro-F1 score: ", ma_score)

