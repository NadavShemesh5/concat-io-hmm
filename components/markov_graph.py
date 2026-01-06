import numpy as np
import networkx as nx
from networkx import DiGraph


class MarkovGraph(DiGraph):
    def __init__(self, n_epochs=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_epochs = n_epochs
        self.order = None

    def train(self, train, validation):
        # TODO: wrap n_epochs, training_loop and learning rate in scheduler class
        self.init_graph(train.vocab_size)
        valid_batch, valid_lens, _ = next(validation)
        it = 0
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            for train_batch, train_lens, final_batch in train:
                self.feed_data(train_batch, train_lens)
                lr = self.compute_learning_rate(it)
                log_prob = self.training_loop(lr)
                print("Curr Batch Perplexity:", np.exp(-log_prob / sum(train_lens)))
                it += 1
                if it % 5 == 0:
                    self.evaluate(valid_batch, valid_lens)
                if final_batch:
                    break

    @staticmethod
    def compute_learning_rate(it):
        return (1.0 + it)**(-0.7)
        # return 1

    def training_loop(self, lr):
        self.backward(lr)
        log_prob = self.forward(lr=lr)
        return log_prob

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

        self.forward()

    def backward(self, lr=0):
        for node in reversed(self.order):
            if node != self.order[0] and lr > 0:
                node.fit_batch(lr)

            node.send_backward_messages()

    def forward(self, lr=0):
        log_prob = None
        for node in self.order:
            if lr > 0:
                log_prob = node.fit_batch(lr)

            node.send_forward_messages()

        return log_prob

    def evaluate(self, batch, lengths, is_train=False):
        self.feed_data(batch, lengths)
        log_prob = self.order[-1].evaluate()
        dataset = "Train" if is_train else "Validation"
        print(f"{dataset} Perplexity:", np.exp(-log_prob / sum(lengths)))

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