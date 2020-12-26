from utils import *
from DeepWalk import DeepWalk
from Node2vec import Node2vec
from Line_1 import Line_1
from NetMF import NetMF
from GraphSage import GraphSage

# Global                                                                          
FILE_NAME = "sample_data.txt"

def main():
    data = read_file(FILE_NAME)
    V, E = process_data(data)

    # Instancing DeepWalk class
    dw = DeepWalk(data)
    dw_result = dw.deep_walk()
    # TODO: visualize the results

    # Instancing Node2vec class
    n2v = Node2vec()
    n2v_result = n2v.main()
    # TODO: visualize the results

    # Instancing Line_1 class
    l1 = Line_1()
    l1_result = l1.main()
    # TODO: visualize the results

    # Instancing NetMF class
    nmf = NetMF()
    nmf_result = nmf.main()
    # TODO: visualize the results

    # Instancing GraphSage class
    gs = GraphSage(data)
    gs_result = gs.graph_sage()
    # TODO: visualize the results


if __name__ == "__main__":
    main()

