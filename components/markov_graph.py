import numpy as np
import networkx as nx
from networkx import DiGraph


class MarkovGraph(DiGraph):
    def __init__(self, n_iter=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = n_iter

    def train(self, data_iterator):
        order = list(nx.topological_sort(self))
        next(data_iterator)
        for _ in range(self.n_iter):
            self.backward(order, True)
            self.forward(order)
            self.evaluate(order, data_iterator)

    @staticmethod
    def backward(order, preprocess=False):
        if preprocess:
            for node in order[:-2]:
                node.send_forward_messages()

        for node in reversed(order[2:]):
            node.fit_batch()
            node.send_backward_messages()

        order[1].fit_batch()

    @staticmethod
    def forward(order):
        for node in order[:-2]:
            node.fit_batch()
            node.send_forward_messages()

        order[-2].fit_batch()

    @staticmethod
    def evaluate(order, data_iterator):
        for node in order[:-2]:
            node.send_forward_messages(choose_max=False)

        order[-1].send_backward_messages()
        log_loss = order[-2].evaluate()
        print("Perplexity:", np.exp(-log_loss / sum(len(x[1]) for x in data_iterator.current_batch)))
