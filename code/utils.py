import os, sys, time, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_file(file_name):
    """
    Read a graph from a txt file. Each line of the txt file is an edge.
    """
    print("Reading file...")
    data = []
    with open(file_name, 'r') as f:
        try:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                values = [int(x.strip()) for x in line.split()]
                data.append(values)
            data = np.array(data)
            print(r"Successfully read %s." % file_name)
            return data
        except:
            print("Failed to read %s." % file_name)

def process_data(data):
    """
    Return the set of vertices and edges.
    :type V: set
    :type E: 2-d np array
    """
    print("Processing data...")
    V = set()
    for edge in data:
        V.add(edge[0])
        V.add(edge[1])
    E = data
    print("Data processing completed.")
    return V, E


if __name__ == "__main__":
    data = read_file("sample_data.txt")
    process_data(data)

