from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Global                                                                          
SEED = 0
EMBEDDING_PATH = "../results/chameleon/Chameleon_NetMF.txt"
LABEL_PATH = "../datasets/chameleon/chameleon_label.txt"

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
    
<<<<<<< HEAD
    def process_labels(self, label_path):
=======
    def read_labels(self, path):
        labels = []
>>>>>>> c905a9719d2db7bb0f684d3fa3c37d047f829773
        all_labels = set()
        # read label
        y = []
        name = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
<<<<<<< HEAD
            for i in range(len(lines)):
                line = lines[i]
                values = [int(x.strip()) for x in line.split()]
                y.append(values[1:])
                name.append(np.array([values[0]]))
                for value in values[1:]:
                    all_labels.add(value)
            y = np.array(y)
            name = np.array(name)
=======
            for line in lines:
                values = list(map(int, line.split()))
                x, *y = values
                labels.append((x, y))
                all_labels.update(y)
>>>>>>> c905a9719d2db7bb0f684d3fa3c37d047f829773

        labels.sort(key = lambda p: p[0])

        # tranform to boolean matrix
<<<<<<< HEAD
        boolean_matrix = np.zeros((len(y),len(all_labels)))
        for i in range(len(y)):
            for value in y[i]:
                boolean_matrix[i][value] = 1
=======
        boolean_matrix = np.zeros((len(labels), len(all_labels)), dtype=np.int)
        for i, p in enumerate(labels):
            boolean_matrix[i, p[1]] = 1
>>>>>>> c905a9719d2db7bb0f684d3fa3c37d047f829773

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
