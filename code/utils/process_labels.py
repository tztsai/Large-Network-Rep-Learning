import numpy as np

LABEL_PATH = "../test/blogcataloglabel.txt"

def process_labels(label_path):
    all_labels = set()
    # read label
    y = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            values = [int(x.strip()) for x in line.split()]
            y.append(values[1:])
            for value in values[1:]:
                all_labels.add(value)
        y = np.array(y)

    # tranform to boolean matrix
    boolean_matrix = np.zeros((len(y),len(all_labels)))
    for i in range(len(y)):
        for value in y[i]:
            boolean_matrix[i][value] = 1

    return boolean_matrix


if __name__ == "__main__":
    process_labels(LABEL_PATH)
                     

