import random
import logging
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
from utils.graph import Graph, read_graph
from DeepWalk import RandomWalk, HLogSoftMax

logger = logging.getLogger('GraphSage')


class GraphSage(nn.Module):
    bs = 64                 # batch size
    lr = config.ALPHA       # learning rate
    
    def __init__(self, graph, load_model=True):
        super.__init__()

        self.N = graph.num_nodes
        self.D = config.D   # embedding dimension

    def loss(self):
        