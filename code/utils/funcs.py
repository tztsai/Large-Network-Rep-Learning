import random
from time import time
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing as mp
from functools import wraps


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


def norm(x):
    return np.linalg.norm(x)


def cos_similarity(x, y):
    return (x@y) / np.sqrt((x@x) * (y@y))


def timer(f):
    @wraps(f)
    def wrapped(*args, **kwds):
        t0 = time()
        result = f(*args, **kwds)
        t1 = time()
        ms = int((t1 - t0) * 1000)
        s, ms = divmod(ms, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        print("Time cost of '%s': %s%s%s%s" %
              (f.__name__,
               f'{h}h' if h else '',
               f'{m}m' if m else '',
               f'{s}s' if s else '',
               f'{ms}ms'))
        return result
    return wrapped


@timer
def test(a):
    for _ in range(a):
        pass
 
test(1000000)
test(10000000000)