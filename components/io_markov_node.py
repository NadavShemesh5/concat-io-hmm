import numpy as np

from components.tools import normalize, timing
from components.markov_node import MarkovNode
from algo import io_baum_welch


class IOMarkovNode(MarkovNode):
    def __init__(self, n_inputs, n_outputs, n_states=4, **kwargs):
        super().__init__(n_states=n_states, **kwargs)
        self.n_inputs = n_inputs
        self.n_tokens = n_outputs

    def initialize_sufficient_statistics(self):
        stats = {
            "trans": np.zeros((self.n_inputs, self.n_states, self.n_states)),
            "obs": np.zeros((self.n_inputs, self.n_states, self.n_tokens)),
        }
        return stats

    def compute_likelihood(self, input_seq, output_seq):
        return self.emit_mat[input_seq, :, output_seq]

    def fit(self, sequence_pair):
        input_seq, output_seq = sequence_pair

        frameprob = self.compute_likelihood(input_seq, output_seq)

        log_prob, fwdlattice, scaling_factors = io_baum_welch.forward(
            self.trans_mat, frameprob, input_seq
        )

        bwdlattice = io_baum_welch.backward(
            self.trans_mat,
            frameprob,
            scaling_factors,
            input_seq,
        )

        posteriors = self.compute_posteriors(fwdlattice, bwdlattice)
        return (
            (input_seq, output_seq, frameprob),
            log_prob,
            posteriors,
            fwdlattice,
            bwdlattice,
        )

    def accumulate_sufficient_statistics(
        self, stats, data_tuple, lattice, posteriors, fwdlattice, bwdlattice
    ):
        input_seq, output_seq, frameprob = lattice

        io_baum_welch.compute_xi_sum(
            fwdlattice, self.trans_mat, bwdlattice, frameprob, input_seq, stats["trans"]
        )

        for k in range(self.n_states):
            np.add.at(stats["obs"][:, k, :], (input_seq, output_seq), posteriors[:, k])

    def m_step(self, stats):
        self.trans_mat = stats["trans"] + self.trans_prior
        normalize(self.trans_mat, axis=2)

        self.emit_mat = stats["obs"] + self.emit_prior
        normalize(self.emit_mat, axis=2)

    def evaluate(self, X):
        log_prob = 0
        for input_seq, output_seq in X:
            frameprob = self.compute_likelihood(input_seq, output_seq)
            l_prob, _, _ = io_baum_welch.forward(self.trans_mat, frameprob, input_seq)
            log_prob += l_prob

        return log_prob

    def perplexity(self, X):
        return np.exp(- self.evaluate(X) / sum(len(x[1]) for x in X))

    def init_parameters(self, X, vocab):
        if self.is_trained:
            return

        init = 1.0 / self.n_states
        self.trans_mat = self.random_state.dirichlet(np.full(self.n_states, init), size=(self.n_inputs, self.n_states))

        self.emit_mat = self.random_state.rand(self.n_inputs, self.n_states, self.n_tokens)
        normalize(self.emit_mat, axis=2)
