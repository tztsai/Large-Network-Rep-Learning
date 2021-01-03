from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from NetMF import NetMF

# Global                                                                          
PATH = "./results/embeddings/"
FILE_NAME = "NetMF.txt"

class NodeClassification():
    def __init__(self):
        self.embedding = None
        self.X = None
        self.y = None


    def read_file(self, file_path):
        self.embedding = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                values = [float(x.strip()) for x in line.split()]
                self.embedding.append(values)
            self.embedding = np.array(self.embedding)

        # construct inputs and labels
        return X, y
    
    def node_classification(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        clf = OneVsRestClassifier(LogisticRegression()).fit(X, y)
        clf.fit(X_train, y_train)
        scores = cross_validate(clf, X_test, y_test, scoring='precision_macro', cv=5)
        return np.mean(scores['test_score'])


if __name__ == "__main__":
    nc = NodeClassification()
    X, y = nc.read_file(PATH+FILE_NAME)
    score = nc.node_classification()

