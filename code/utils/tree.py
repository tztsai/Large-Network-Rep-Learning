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

        def __lt__(self, other):
            return self.value < other.value

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
    def __init__(self):
        super().__init__()
        self._leaves = {}
        self._code = {}

    def build_from_nodes(self, weighted_nodes):
        """
        Build the Huffman tree from a sequence of weighted nodes so that the higher the weight,
        the shorter the path from root to the node.

        Args:
            weighted_nodes: a (value, weight) sequence
        """
        pq = PriorityQueue()

        for v, w in weighted_nodes:
            node = Tree.Node((v, w))
            pq.put((w, node))
            self._leaves[v] = node

        while not pq.empty():
            w1, n1 = pq.get()
            if pq.empty():
                self.root = n1
            else:
                w2, n2 = pq.get()
                w = w1 + w2
                n = Tree.Node((-1, w))
                n.add_child(n1)
                n.add_child(n2)
                pq.put((w, n))

    def build_from_graph(self, graph: Graph):
        self.build_from_nodes((n, sum(graph.neighbors[n].values()))
                              for n in graph.nodes)

    def code(self, value):
        if value in self._code:
            return self._code[value]

        c = ''
        node = self._leaves[value]
        while node is not self.root:
            parent = node.parent
            c = str(parent.children.index(node)) + c
            node = parent

        self._code[value] = c
        self._leaves.pop(value)  # no longer needed
        return c


if __name__ == '__main__':
    from graph import read_graph
    g = read_graph("small.txt")
    print([(n, sum(g.neighbors[n].values())) for n in g.nodes])
    ht = HuffmanTree()
    ht.build_from_graph(g)
    ht.print()
    print(ht.code(1), ht.code(2), ht.code(3), ht.code(1))

    ht2 = HuffmanTree()
    ht2.build_from_nodes(enumerate([2,4,1,6,5,7,3,0]))
    ht2.print()
    print([ht2.code(i) for i in range(8)])
