import numpy as np
import networkx as nx
from networkx import DiGraph


class MarkovGraph(DiGraph):
    def __init__(self, n_epochs=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_epochs = n_epochs
        self.order = None

    def train(self, train, validation):
        # TODO: wrap n_epochs, training_loop scheduling and multi_sample in scheduler class
        self.init_graph(train.vocab_size)
        valid_batch, valid_lens, _ = next(validation)
        for epoch in range(self.n_epochs):
            for train_batch, train_lens, final_batch in train:
                self.feed_data(train_batch, train_lens)
                self.training_loop()
                if final_batch:
                    break

            # self.evaluate(train_batch, train_lens, is_train=True)
            self.evaluate(valid_batch, valid_lens)

    def training_loop(self):
        self.backward()
        self.forward()

    def feed_data(self, data, lengths):
        for node in self.nodes:
            node.reset_batches()
            node.batch_lengths = lengths

        for node in self.order:
            if node.input_index == -1:
                break

            input_data = data[:, node.input_index + 1, :]
            node.forward_batches = [input_data]

        for node in reversed(self.order):
            if node.output_index == -1:
                break

            output_data = data[:, node.output_index, :]
            node.backward_batches = [output_data]

        self.forward(train=False)

    def backward(self):
        for node in reversed(self.order):
            node.fit_batch()
            node.send_backward_messages()

    def forward(self, train=True):
        for node in self.order:
            if train:
                node.fit_batch()

            node.send_forward_messages()

    def evaluate(self, batch, lengths, is_train=False):
        self.feed_data(batch, lengths)
        log_loss = self.order[-1].evaluate()
        dataset = "Train" if is_train else "Validation"
        print(f"{dataset} Perplexity:", np.exp(-log_loss / sum(lengths)))

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