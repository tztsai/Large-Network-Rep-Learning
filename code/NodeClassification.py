from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from NetMF import NetMF
from utils.process_labels import *
import numpy as np

# Global                                                                          
EMBEDDING_PATH = "./test/blogcatalog_NetMF_embedding.txt"
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
        # read label
        self.y = process_labels(label_path)
        return self.X, self.y
    
    def node_classification(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # one-vs-rest logistic regression
        clf = OneVsRestClassifier(LogisticRegression(random_state=0)).fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # use max-vote
        # to do
        return score

    def max_vote(self):
        pass


if __name__ == "__main__":
    nc = NodeClassification()
    X, y = nc.read_file(EMBEDDING_PATH, LABEL_PATH)
    score = nc.node_classification(X, y)
    print("Accurancy for node classification: ", score)

