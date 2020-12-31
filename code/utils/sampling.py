import numpy as np

class alias:
    def __init__(self, problist):
        self.nodenumber = len(problist)
        self.alias = np.zeros(self.nodenumber)
        self.prob = np.zeros(self.nodenumber)
        self.setup(problist)
    
    def setup(self, problist):
        less1 = []
        more1 = []
        
        for idnum, value in enumerate(problist):
            self.prob[idnum] = self.nodenumber * value
            if self.prob[idnum] >= 1:
                more1.append(idnum)
            else:
                less1.append(idnum)
        
        while len(less1) > 0 and len(more1) > 0:
            More = more1.pop()
            Less = less1.pop()
            
            self.alias[Less] = More
            self.prob[More] = self.prob[More] + self.prob[Less] - 1
            
            if self.prob[More] >= 1:
                more1.append(More)
            else:
                less1.append(More)
    
    def draw(self):
        pick = np.random.rand()
        pnt = int(np.floor(pick * self.nodenumber))
        
        pick = np.random.rand()
        if pick > self.prob[pnt]:
            return int(self.alias[pnt])
        else:
            return int(pnt)
            
    def sample(self, number):
        result = []
        for i in range(number):
            sampleres = self.draw()
            result.append(sampleres)
        return result


    