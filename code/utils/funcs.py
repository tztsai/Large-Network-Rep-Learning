import random
import numpy as np
import torch
from tqdm import tqdm


def pbar(iterable, **kwds):
    """process bar visualization"""
    return tqdm(iterable, bar_format='\t{l_bar}{bar:20}{r_bar}', **kwds)
