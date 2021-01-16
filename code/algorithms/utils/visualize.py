""" Visualization Utilities """

import numpy as np
import matplotlib.pyplot as plt
from .funcs import cos_sim


def plot_loss(losses, title='Loss', xlabel='Epoch', ylabel='Loss'):
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_adj(embeddings):
    n = len(embeddings)
    adj = np.zeros((n, n))
    for i, zi in enumerate(embeddings):
        for j, zj in enumerate(embeddings):
            adj[i, j] = cos_sim(zi, zj)
    plt.imshow(adj)
    plt.colorbar()
    plt.show()
    
    
    
if __name__ == "__main__":
    A = np.random.rand(5, 5)
    plot_adj(A)