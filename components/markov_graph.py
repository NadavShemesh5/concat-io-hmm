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
        valid_batch, valid_lens = next(validation)
        train_batch, train_lens = next(train)
        multi_sample = 32
        train_choose_max = True
        train_multi_sample = multi_sample if not train_choose_max else 1
        valid_choose_max = True
        valid_multi_sample = multi_sample if not valid_choose_max else 1
        for _ in range(self.n_iter):
            self.feed_data(train_batch, train_lens, train_choose_max, multi_sample=train_multi_sample)
            self.training_loop(train_choose_max)

            self.evaluate(train_batch, train_lens, valid_choose_max, multi_sample=valid_multi_sample, is_train=True)
            # self.evaluate(valid_batch, valid_lens)

    def training_loop(self, choose_max):
        self.backward(choose_max)
        self.forward(choose_max)

    def feed_data(self, data, lengths, choose_max, multi_sample=1):
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

        self.forward(choose_max, train=False)

    @staticmethod
    def tile_batch(batch, multi_sample):
        if multi_sample == 1:
            return [batch]

        if batch.ndim == 2:
            return [np.tile(batch, (multi_sample, 1))]
        else:
            return [np.tile(batch, multi_sample)]

    def backward(self, choose_max):
        for node in reversed(self.order):
            node.fit_batch()
            node.send_backward_messages(choose_max)

    def forward(self, choose_max, train=True):
        for node in self.order:
            if train:
                node.fit_batch()

            node.send_forward_messages(choose_max)

    def evaluate(self, batch, lengths, choose_max, multi_sample=1, is_train=False):
        assert not (multi_sample > 1 and choose_max), "Cannot evaluate with both multi-sample and choose-max."
        self.feed_data(batch, lengths, choose_max, multi_sample=multi_sample)
        log_loss = self.order[-1].evaluate()
        dataset = "Train" if is_train else "Validation"
        print(f"{dataset} Perplexity:", np.exp(-log_loss / (sum(lengths) * multi_sample)))

    def init_graph(self, vocab_size):
        for node in self.nodes:
            if not node.n_inputs:
                node.n_inputs = vocab_size

            node.init_matrices(vocab_size)

        print("Total Parameters:", self.num_of_parameters())
        self.order = list(nx.topological_sort(self))

    def num_of_parameters(self):
        total_params = 0
        for node in self.nodes:
            total_params += node.num_of_parameters()

        return total_params