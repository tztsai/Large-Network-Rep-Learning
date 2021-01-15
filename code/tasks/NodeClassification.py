from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import sys

# Global                                                                          
SEED = 0
EMBEDDING_PATH = sys.argv[1]
LABEL_PATH = sys.argv[2]

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
            self.X = np.array(sorted(self.X, key=(lambda x:x[0]), reverse=False))
        # read labels
        self.y = self.process_labels(label_path)
        self.y = np.array(sorted(self.y, key=(lambda x:x[0]), reverse=False))
        return self.X, self.y
    
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
    ma_score, mi_score = nc.node_classification(X[:,1:], y[:,1:])
    print("Accurancy for node classification: ")
    print("Micro-F1 score: ", mi_score)
    print("Macro-F1 score: ", ma_score)
