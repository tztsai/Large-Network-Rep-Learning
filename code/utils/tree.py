class Tree:
    """A probabilistic tree model."""

    class Node:
        def __init__(self, value=None, parent=None):
            self.value = value
            self.parent = parent

    def __init__(self, max_branches=2):
        self.root = Tree.Node()
        self.b = max_branches

    # def build(self, ):
