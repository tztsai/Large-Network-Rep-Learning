import random
import numpy as np


class alias:
    def __init__(self, problist):
        self.num_nodes = len(problist)
        self.alias = np.zeros(self.num_nodes, dtype=np.int)
        self.prob = np.zeros(self.num_nodes)
        self.setup(problist)
    
    def setup(self, problist):
        Small = []
        Big = []
        
        for i, p in enumerate(problist):
            self.prob[i] = self.num_nodes * p
            if self.prob[i] >= 1:
                Big.append(i)
            else:
                Small.append(i)
        
        while Small and Big:
            big = Big.pop()
            small = Small.pop()
            
            self.alias[small] = big
            self.prob[big] = self.prob[big] + self.prob[small] - 1
            
            if self.prob[big] >= 1:
                Big.append(big)
            else:
                Small.append(big)
    
    def draw(self):
        i = random.randrange(self.num_nodes)
        return i if random.random() < self.prob[i] else self.alias[i]
            
    def sample(self, size):
        return [self.draw() for _ in range(size)]
