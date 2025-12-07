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
        if input_seq.ndim == 1:
            trans_dynamic = self.trans_mat[input_seq]
        else:
            trans_dynamic = np.einsum("tk, kij -> tij", input_seq, self.trans_mat)

        if input_seq.ndim == 1 and output_seq.ndim == 1:
            frameprob = self.emit_mat[input_seq, :, output_seq]
        elif input_seq.ndim == 1:
            selected_emit = self.emit_mat[input_seq]
            frameprob = np.einsum("tsy, ty -> ts", selected_emit, output_seq)
        elif output_seq.ndim == 1:
            relevant_emits = self.emit_mat[:, :, output_seq].transpose(2, 0, 1)
            frameprob = np.einsum("tks, tk -> ts", relevant_emits, input_seq)
        else:
            lik_given_u = np.einsum("ksy, ty -> tks", self.emit_mat, output_seq)
            frameprob = np.einsum("tks, tk -> ts", lik_given_u, input_seq)

        return frameprob, trans_dynamic

    def fit(self, sequence_pair):
        input_seq, output_seq = sequence_pair
        frameprob, trans_dynamic = self.compute_likelihood(input_seq, output_seq)
        log_prob, fwdlattice, scaling_factors = io_baum_welch.forward(trans_dynamic, frameprob)
        bwdlattice = io_baum_welch.backward(trans_dynamic, frameprob, scaling_factors)
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
        io_baum_welch.accumulate_trans(
            fwdlattice,
            bwdlattice,
            self.trans_mat,
            frameprob,
            input_seq,
            stats["trans"],
        )
        io_baum_welch.accumulate_obs(
            posteriors,
            input_seq,
            output_seq,
            stats["obs"],
        )

    def m_step(self, stats):
        self.trans_mat = stats["trans"] + self.trans_prior
        normalize(self.trans_mat, axis=2)

        self.emit_mat = stats["obs"] + self.emit_prior
        normalize(self.emit_mat, axis=2)

    def evaluate(self, X):
        log_prob = 0
        for input_seq, output_seq in X:
            frameprob, trans_dynamic = self.compute_likelihood(input_seq, output_seq)
            l_prob, _, _ = io_baum_welch.forward(trans_dynamic, frameprob)
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
