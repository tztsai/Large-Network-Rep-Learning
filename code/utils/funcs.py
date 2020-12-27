import random


def rand_perm(n):
    if type(n) is int:
        n = range(n)
    return random.sample(n, len(n))
