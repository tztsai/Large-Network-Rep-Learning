import numpy as np
from NetMF import NetMF

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
    
    def main(self):
        pass

if __name__ == "__main__":
    lp = LinkPrediction()
    lp.read_file(PATH+FILE_NAME)
    lp.main()
