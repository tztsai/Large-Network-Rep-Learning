import random
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing as mp


def pbar(iterable, **kwds):
    """Process bar visualization."""
    return tqdm(iterable, bar_format='\t{l_bar}{bar:20}{r_bar}', **kwds)

def pmap(func, iterable):
    """Parallely map a function to a sequence."""
    with mp.Pool(mp.cpu_count()) as pool:
        return pool.map(func, iterable)

def split_array(a: np.ndarray, m):
    """Evenly split an array into m parts."""
    l = len(a) // m
    return [a[i*l : (i+1)*l] for i in range(m)]