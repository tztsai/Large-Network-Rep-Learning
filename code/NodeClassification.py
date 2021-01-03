from DeepWalk import DeepWalk
from Node2vec import Node2vec
from Line_1 import Line_1
from NetMF import NetMF
from GraphSage import GraphSage

# Global                                                                          
FILE_NAME = "sample_data.txt"

class NodeClassification():
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
    
    def main(self):
        pass

    def cross_validation(self):
        pass

if __name__ == "__main__":
    nc = NodeClassification()
    nc.read_file(PATH+FILE_NAME)
    nc.main()

