""" Visualization Utilities """

import matplotlib.pyplot as plt


def plot_loss(losses, title='Loss', xlabel='Epoch', ylabel='Loss'):
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()