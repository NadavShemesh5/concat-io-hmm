import numpy as np
from sklearn.utils.validation import check_random_state

from components.tools import (
    normalize,
    timing,
    viterbi_backward,
    viterbi_forward,
    greedy_forward,
    greedy_backward,
)
from algo import io_baum_welch


class IOMarkovNode:
    def __init__(
        self,
        graph,
        n_states,
        n_inputs=None,
        start_prior=1e-6,
        trans_prior=1e-6,
        emit_prior=1e-6,
        random_state=1,
        input_index=-1,
        output_index=-1,
    ):
        self.graph = graph
        self.n_inputs = n_inputs
        self.n_states = n_states
        self.start_prior = start_prior
        self.trans_prior = trans_prior
        self.emit_prior = emit_prior
        self.input_index = input_index
        self.output_index = output_index
        self.random_state = check_random_state(random_state)
        self.is_initialized = False

        self.start_mat = None
        self.emit_mats = []
        self.trans_mat = None
        self.backward_batches = []
        self.parent_sending_order = []
        self.forward_batches = []
        self.batch_lengths = []

    @timing
    def fit_batch(self):
        self.accumulate_forward()
        stats = self.init_trans_optimizer()
        input_batch = self.forward_batches[0]
        if not self.children:
            output_batch = self.backward_batches[0]
            stats = self.init_emit_optimizer(stats, 0)
            self.fit_tensor_batch(input_batch, output_batch, stats, 0)
            self.optimize_emit(stats, 0)

        for idx in range(len(self.children)):
            child = self.children[idx]
            batch_idx = self.parent_sending_order.index(child)
            output_batch = self.backward_batches[batch_idx]
            stats = self.init_emit_optimizer(stats, idx)
            self.fit_tensor_batch(input_batch, output_batch, stats, idx)
            self.optimize_emit(stats, idx)

        self.optimize_trans(stats)

    def init_trans_optimizer(self):
        stats = {
            "trans": np.zeros_like(self.trans_mat, dtype=np.float64),
            "start": np.zeros_like(self.start_mat, dtype=np.float64),
        }
        return stats

    def init_emit_optimizer(self, stats, idx):
        stats["emit"] = np.zeros_like(self.emit_mats[idx], dtype=np.float64)
        return stats

    def optimize_trans(self, stats):
        stats["start"] += self.start_prior
        normalize(stats["start"], axis=0)
        self.start_mat = stats["start"].astype(np.float32)

        stats["trans"] += self.trans_prior
        normalize(stats["trans"], axis=2)
        self.trans_mat = stats["trans"].astype(np.float32)

    def optimize_emit(self, stats, idx):
        stats["emit"] += self.emit_prior
        normalize(stats["emit"], axis=2)
        self.emit_mats[idx] = stats["emit"].astype(np.float32)

    def fit_tensor_batch(self, input_seqs, output_seqs, stats, idx):
        total_log_prob, fwd, scaling = io_baum_welch.forward(
            self.trans_mat,
            self.emit_mats[idx],
            self.start_mat,
            input_seqs,
            output_seqs,
            self.batch_lengths,
        )
        bwd = io_baum_welch.backward(
            self.trans_mat,
            self.emit_mats[idx],
            scaling,
            input_seqs,
            output_seqs,
            self.batch_lengths,
        )
        posteriors = self.compute_posteriors(fwd, bwd)
        io_baum_welch.compute_xi_sum(
            fwd,
            self.trans_mat,
            bwd,
            self.emit_mats[idx],
            input_seqs,
            output_seqs,
            self.batch_lengths,
            stats["trans"],
        )

        start_tokens = input_seqs[:, 0]
        start_posteriors = posteriors[:, 0, :]
        np.add.at(stats["start"].T, start_tokens, start_posteriors)

        max_len = input_seqs.shape[1]
        mask = np.arange(max_len) < self.batch_lengths[:, None]
        valid_in = input_seqs[mask]
        valid_out = output_seqs[mask]
        valid_post = posteriors[mask]
        np.add.at(stats["emit"].transpose(0, 2, 1), (valid_in, valid_out), valid_post)

    @staticmethod
    def compute_posteriors(fwd, bwd):
        posteriors = fwd * bwd
        normalize(posteriors, axis=2)
        return posteriors

    @timing
    def send_forward_messages(self, choose_max):
        self.accumulate_forward()
        input_batch = self.forward_batches[0]
        for child, emit in zip(self.children, self.emit_mats):
            sample_func = viterbi_forward if choose_max else greedy_forward
            forward_sample = sample_func(
                input_batch, self.start_mat.T, self.trans_mat, emit
            )
            child.forward_batches.append(forward_sample)

        if self.children:
            self.backward_batches = []

    @timing
    def send_backward_messages(self, choose_max):
        if not self.parents:
            return

        evidence_pairs = []
        if not self.children:
            batch = self.backward_batches[0]
            emit = self.emit_mats[0]
            evidence_pairs.append((batch, emit))
        else:
            for child, batch in zip(self.parent_sending_order, self.backward_batches):
                emit = self.emit_mats[self.children.index(child)]
                evidence_pairs.append((batch, emit))

        target_batch, emit_mat = evidence_pairs[0]
        sample_func = viterbi_backward if choose_max else greedy_backward
        backward_sample = sample_func(
            target_batch,
            self.start_mat.T,
            self.trans_mat,
            emit_mat
        )

        dims = tuple([self.n_inputs] * len(self.parents))
        unraveled_samples = np.unravel_index(backward_sample, dims)
        for i, parent in enumerate(self.parents):
            parent.backward_batches.append(unraveled_samples[i])
            parent.parent_sending_order.append(self)

        if self.parents:
            self.forward_batches = []

    def accumulate_forward(self):
        if len(self.forward_batches) == 1:
            return

        dims = tuple([self.n_inputs] * len(self.forward_batches))
        combined_batch = np.ravel_multi_index(self.forward_batches, dims)
        self.forward_batches = [combined_batch]

    def evaluate(self):
        self.accumulate_forward()
        input_seqs = self.forward_batches[0]
        output_seqs = self.backward_batches[0]
        total_log_prob, _, _ = io_baum_welch.forward(
            self.trans_mat,
            self.emit_mats[0],
            self.start_mat,
            input_seqs,
            output_seqs,
            self.batch_lengths,
        )
        return total_log_prob

    def init_matrices(self, output_dimension=None):
        if self.is_initialized:
            return

        input_dimension = int(np.prod([self.n_inputs for _ in range(len(self.parents))])) if self.parents else self.n_inputs
        init = 1.0 / self.n_states
        self.trans_mat = self.random_state.dirichlet(np.full(self.n_states, init), size=(input_dimension, self.n_states)).astype(np.float32)
        self.start_mat = self.random_state.dirichlet(np.full(self.n_states, init), size=input_dimension).T.astype(np.float32)

        child_dimensions = [child.n_inputs for child in self.children] or [output_dimension]
        for dim in child_dimensions:
            emit_mat = self.random_state.rand(input_dimension, self.n_states, dim).astype(np.float32)
            normalize(emit_mat, axis=2)
            self.emit_mats.append(emit_mat)

        self.is_initialized = True

    def reset_batches(self):
        self.forward_batches = []
        self.backward_batches = []
        self.parent_sending_order = []
        self.batch_lengths = []

    def num_of_parameters(self):
        total_params = 0
        total_params += sum(emit.size for emit in self.emit_mats)
        if self.n_states > 1:
            total_params += self.start_mat.size
            total_params += self.trans_mat.size

        return total_params

    def add_child(self, child):
        self.graph.add_edge(self, child)

    @property
    def children(self):
        return list(self.graph.successors(self))

    @property
    def parents(self):
        return list(self.graph.predecessors(self))