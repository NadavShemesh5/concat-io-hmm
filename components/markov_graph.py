import numpy as np
import networkx as nx
from networkx import DiGraph


class MarkovGraph(DiGraph):
    def __init__(self, n_iter=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = n_iter
        self.order = None

    def train(self, train, validation):
        # TODO: wrap n_iter, training_loop scheduling and multi_sample in scheduler class
        self.init_graph(train.vocab_size)
        for (train_batch, train_lens), (valid_batch, valid_lens) in zip(train, validation):
            for _ in range(self.n_iter):
                self.feed_data(train_batch, train_lens, multi_sample=1, choose_max=True)
                for _ in range(1):
                    self.training_loop()

                self.evaluate(train_batch, train_lens, is_train=True)
                # self.evaluate(valid_batch, valid_lens)


    def training_loop(self):
        self.backward()
        self.forward()

    def feed_data(self, data, lengths, multi_sample=1, choose_max=False):
        lengths =  self.tile_batch(lengths, multi_sample)[0]
        for node in self.nodes:
            node.reset_batches()
            node.batch_lengths = lengths

        for node in self.order:
            if node.input_index == -1:
                break

            input_data = data[:, node.input_index + 1, :]
            node.forward_batches = self.tile_batch(input_data, multi_sample)

        for node in reversed(self.order):
            if node.output_index == -1:
                break

            output_data = data[:, node.output_index, :]
            node.backward_batches = self.tile_batch(output_data, multi_sample)

        self.forward(train=False, choose_max=choose_max)

    @staticmethod
    def tile_batch(batch, multi_sample):
        if multi_sample == 1:
            return [batch]

        if batch.ndim == 2:
            return [np.tile(batch, (multi_sample, 1))]
        else:
            return [np.tile(batch, multi_sample)]

    def backward(self):
        for node in reversed(self.order):
            node.fit_batch()
            node.send_backward_messages()

    def forward(self, train=True, choose_max=False):
        for node in self.order:
            if train:
                node.fit_batch()

            node.send_forward_messages(choose_max=choose_max)

    def evaluate(self, batch, lengths, is_train=False, choose_max=True):
        self.feed_data(batch, lengths, choose_max=choose_max)
        log_loss = self.order[-1].evaluate()
        dataset = "Train" if is_train else "Validation"
        print(f"{dataset} Perplexity:", np.exp(-log_loss / sum(lengths)))

    def init_graph(self, vocab_size):
        for node in self.nodes:
            if not node.n_inputs:
                node.n_inputs = vocab_size

            node.init_matrices(vocab_size)

        self.order = list(nx.topological_sort(self))
