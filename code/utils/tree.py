class Tree:
    branches = 2

    class Node:
        def __init__(self, value=None, parent=None):
            self.value = value
            self.parent = parent
            self.children = []

        def add_child(self, value=None):
            if len(self.children) >= Tree.branches:
                raise RuntimeError("Reached max number of children.")
            child = Tree.Node(value, parent=self)
            self.children.append(child)
            return child

    def __init__(self, values=()):
        """
        Initialize the tree and its root node;
        build the tree from the values sequence if it is given.
        """
        self.root = Tree.Node()
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


if __name__ == '__main__':
    t = Tree(range(15))
    t.print()
