from queue import PriorityQueue
from graph import Graph


class Tree:
    branches = 2

    class Node:
        def __init__(self, value=None, parent=None):
            self.value = value
            self.parent = parent
            self.children = []

        def add_child(self, child):
            if len(self.children) >= Tree.branches:
                raise RuntimeError("Reached max number of children.")
            if not isinstance(child, Tree.Node):
                child = Tree.Node(child)
            self.children.append(child)
            child.parent = self
            return child

    def __init__(self):
        self.root = Tree.Node()

    def build_from_list(self, values):
        values = iter(values)
        to_visit = [self.root]
        while True:
            try:
                node = to_visit.pop(0)
                if node is self.root:
                    node.value = next(values)
                for _ in range(self.branches):
                    child = node.add_child(next(values))
                    to_visit.append(child)
            except StopIteration:
                break

    def print(self):
        """Print the topology of the tree."""
        visit_list = [self.root]
        visit_depth = [0]

        while len(visit_list) != 0:
            cur_node = visit_list[0]
            cur_depth = visit_depth[0]

            print("\t" * (cur_depth + 1) + str(cur_node.value))
            visit_list = visit_list[1:]
            visit_list = cur_node.children + visit_list
            visit_depth = visit_depth[1:]
            visit_depth = [cur_depth + 1] * len(cur_node.children) + visit_depth


class HuffmanTree(Tree):
    def build_from_nodes(self, weighted_nodes):
        """
        Build the Huffman tree from a sequence of weighted nodes so that the higher the weight,
        the shorter the path from root to the node.

        Args:
            weighted_nodes: a (node, weight) sequence
        """
        pq = PriorityQueue()

        for node, weight in weighted_nodes:
            pq.put((weight, node))

        while not pq.empty():
            _, n1 = pq.get()
            if pq.empty():
                self.root = n1
            else:
                _, n2 = pq.get()


    def build_from_graph(self, graph: Graph):
        self.build_from_nodes((n, len(graph.neighbors[n]) for n in graph.nodes))



if __name__ == '__main__':
    pass
