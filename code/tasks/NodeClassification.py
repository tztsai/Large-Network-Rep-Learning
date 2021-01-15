from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Global                                                                          
SEED = 0
# EMBEDDING_PATH = "./test/blogcatalog_NetMF_embedding_decoded.txt"
# EMBEDDING_PATH = "./test/blogcatalogedge_deepwalk.txt"
# EMBEDDING_PATH = "./test/node2vec_blogcatalog_sort.embed"
# EMBEDDING_PATH = "./test/Line1embd_second-order.txt"
EMBEDDING_PATH = "results/blogcatalog/blogcatalog_graphsage.txt"
LABEL_PATH = "datasets/blogcatalog/blogcataloglabel.txt"

class NodeClassification:
    def __init__(self, embedding_path, labels_path):
        self.X = self.read_embedding(embedding_path)
        self.y = self.read_labels(labels_path)
    
    def read_embedding(self, path):
        with open(path, 'r') as f:
            Z = [list(map(float, line.split()))
                      for line in f.readlines()]
            Z.sort(key = lambda x: x[0])
        Z = np.array(Z)[:, 1:]
        print('Successfully read embedding array from', path)
        return Z
    
    def read_labels(self, path):
        labels = []
        all_labels = set()
        
        # read labels
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                values = list(map(int, line.split()))
                x, *y = values
                labels.append((x, y))
                all_labels.update(y)

        labels.sort(key = lambda p: p[0])

        # tranform to boolean matrix
        boolean_matrix = np.zeros((len(labels), len(all_labels)), dtype=np.int)
        for i, p in enumerate(labels):
            boolean_matrix[i, p[1]] = 1

        print('Successfully read labels from', path)
        return boolean_matrix

    def evaluate(self):
        # one-vs-rest logistic regression
        clf = OneVsRestClassifier(LogisticRegression(random_state=SEED))
        cv = 5
        # report Micro-F1 and Macro-F1 scores
        res = clf.fit(self.X, self.y)
        ma_scores = cross_val_score(clf, self.X, self.y, cv=cv, scoring='f1_macro')
        mi_scores = cross_val_score(clf, self.X, self.y, cv=cv, scoring='f1_micro')
        return np.mean(ma_scores), np.mean(mi_scores)


if __name__ == "__main__":
    nc = NodeClassification(EMBEDDING_PATH, LABEL_PATH)
    ma_score, mi_score = nc.evaluate()
    print("Accurancy for node classification: ")
    print("Micro-F1 score: ", mi_score)
    print("Macro-F1 score: ", ma_score)
