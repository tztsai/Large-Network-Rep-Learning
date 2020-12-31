import networkx as nx
import numpy as np
from utils.sampling import alias

class txtGreader:
    def __init__(self, filename, direct = False, weighted = False):
        self.filename = filename
        self.direct = direct
        
        if weighted:
            self.graph = nx.read_edgelist(self.filename, nodetype = str, data = [("weight", float)], create_using = nx.DiGraph())
        else:
            self.graph = nx.read_edgelist(self.filename, nodetype = str, create_using = nx.DiGraph())
        
        self.edge_num = self.graph.number_of_edges()
        self.node_num = self.graph.number_of_nodes()
        
        self.edgewithweight = self.graph.edges(data=True)
        self.node = self.graph.nodes()
        
        
        self.nodedict = {}
        self.indexdict = {}
        for index, node in enumerate(self.node):
            self.nodedict[index] = node
            self.indexdict[node] = index
        
        self.edge = []
        if weighted:
            self.edge_dist = np.zeros(self.edge_num)
            for index, (pt1, pt2, weight) in enumerate(self.edgewithweight):
                self.edge.append((self.indexdict[pt1], self.indexdict[pt2]))
                self.edge_dist[index] = weight["weight"]
        else:
            self.edge_dist = np.ones(self.edge_num)
            for index, (pt1, pt2, _) in enumerate(self.edgewithweight):
                self.edge.append((self.indexdict[pt1], self.indexdict[pt2]))
        
        self.edge_dist /= np.sum(self.edge_dist)
        
        self.node_degree = np.zeros(self.node_num)
        for index in range(self.node_num):
            node = self.nodedict[index]
            if weighted:
                self.node_degree[index] = self.graph.degree(node, weight="weight")
            else:
                self.node_degree[index] = self.graph.degree(node)
        
        self.node_dist = np.power(self.node_degree, 0.75)
        self.node_dist /= np.sum(self.node_dist)
        
        self.edge_sampling = alias(self.edge_dist)
        self.node_sampling = alias(self.node_dist)
        
    def getbatch(self, batch_size, K):
        #edge sampling
        result = self.edge_sampling.sample(batch_size)
        
        vertex_i = []
        vertex_j = []
        weight = []
        
        for i in result:
            edge = self.edge[i]
            
            
            if np.random.rand() > 0.5:      
                edge = (edge[1], edge[0])
                
            vertex_i.append(edge[0])
            vertex_j.append(edge[1])
            weight.append(1)
            
            #negative sampling
            for j in range(K):
                
                flag = True
                while flag:
                    
                    node = self.node_sampling.draw()
                    flag = self.graph.has_edge(self.nodedict[node], self.nodedict[edge[0]])
                   
                vertex_i.append(edge[0])
                vertex_j.append(node)
                weight.append(-1)
        
        return vertex_i, vertex_j, weight

