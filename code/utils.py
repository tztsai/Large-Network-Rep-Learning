import os, sys, time, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_file(file_name):
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


if __name__ == "__main__":
    data = read_file("sample_data.txt")

