from queue import PriorityQueue


class HuffmanTree:

    def __init__(self, weights):
        n = len(weights)

        self.weight = [0] * n * 2
        self.parent = [0] * n * 2
        self.code = [0] * n * 2

        for i in range(n):
            self.weight[i] = weights[i]

        pq = PriorityQueue()

        for i, w in enumerate(weights):
            pq.put((w, i))

        p = len(weights)
        while not pq.empty():
            w1, i1 = pq.get()
            if pq.empty():  # root node
                self.parent[i1] = -1
            else:
                w2, i2 = pq.get()
                w = w1 + w2
                self.parent[i1] = self.parent[i2] = p
                self.weight[p] = w
                self.code[i1] = 0
                self.code[i2] = 1
                pq.put((w, p))
                p += 1

    def encode(self, n):
        code = ''
        while self.parent[n] >= 0:
            code = str(self.code[n]) + code
            n = self.parent[n]
        return code


if __name__ == '__main__':
    weights = [2,4,1,6,5,7,3,0]
    h = HuffmanTree(weights)
    print(h.weight)
    print(h.parent)
    print(h.code)
    print([h.encode(i) for i in range(4)])
