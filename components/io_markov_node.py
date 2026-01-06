import numpy as np
from sklearn.utils.validation import check_random_state
from scipy.stats import mode

from components.tools import normalize, timing
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
        dropout_rate=0.0,
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
        self.dropout_rate = dropout_rate

        self.start_mat = None
        self.emit_mats = []
        self.trans_mat = None
        self.backward_batches = []
        self.parent_sending_order = []
        self.forward_batches = []
        self.batch_lengths = []

    @timing
    def fit_batch(self, lr):
        self.forward_batches = [self.accumulate_messages(self.forward_batches)]
        stats = self.init_trans_optimizer()
        input_batch = self.forward_batches[0]
        log_prob = 0.0
        if not self.children:
            output_batch = self.backward_batches[0]
            stats = self.init_emit_optimizer(stats, 0)
            log_prob += self.fit_tensor_batch(input_batch, output_batch, stats, 0)
            self.optimize_emit(stats, 0, lr)

        for idx in range(len(self.children)):
            child = self.children[idx]
            batch_idx = self.parent_sending_order.index(child)
            output_batch = self.backward_batches[batch_idx]
            stats = self.init_emit_optimizer(stats, idx)
            log_prob += self.fit_tensor_batch(input_batch, output_batch, stats, idx)
            self.optimize_emit(stats, idx, lr)

        self.optimize_trans(stats, lr)
        return log_prob

    def init_trans_optimizer(self):
        stats = {
            "trans": np.zeros_like(self.trans_mat, dtype=np.float64),
            "start": np.zeros_like(self.start_mat, dtype=np.float64),
        }
        return stats

    def init_emit_optimizer(self, stats, idx):
        stats["emit"] = np.zeros_like(self.emit_mats[idx], dtype=np.float64)
        return stats

    def optimize_trans(self, stats, lr):
        stats["start"] += self.start_prior
        normalize(stats["start"], axis=0)
        self.start_mat = self.update_according_to_lr(self.start_mat, stats["start"], lr)

        stats["trans"] += self.trans_prior
        normalize(stats["trans"], axis=2)
        self.trans_mat = self.update_according_to_lr(self.trans_mat, stats["trans"], lr)

    def optimize_emit(self, stats, idx, lr):
        stats["emit"] += self.emit_prior
        normalize(stats["emit"], axis=2)
        self.emit_mats[idx] = self.update_according_to_lr(self.emit_mats[idx], stats["emit"], lr)

    @staticmethod
    def update_according_to_lr(old, new, lr):
        return (1 - lr) * old + lr * new

    def fit_tensor_batch(self, input_seqs, output_seqs, stats, idx):
        emit = self.dropout_emit(self.emit_mats[idx])
        log_prob, fwd, scaling = io_baum_welch.forward(
            self.trans_mat,
            emit,
            self.start_mat,
            input_seqs,
            output_seqs,
            self.batch_lengths,
        )
        bwd = io_baum_welch.backward(
            self.trans_mat,
            emit,
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
            emit,
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
        return log_prob

    @staticmethod
    def compute_posteriors(fwd, bwd):
        posteriors = fwd * bwd
        normalize(posteriors, axis=2)
        return posteriors

    @timing
    def send_forward_messages(self):
        if not self.children:
            return

        self.forward_batches = [self.accumulate_messages(self.forward_batches)]
        input_batch = self.forward_batches[0]
        for child, emit in zip(self.children, self.emit_mats):
            forward_sample = io_baum_welch.posterior_predict_output(
                self.trans_mat, emit, self.start_mat.T, input_batch, self.batch_lengths
            )
            child.forward_batches.append(forward_sample)

        self.backward_batches = []

    @timing
    def send_backward_messages(self):
        if not self.parents:
            return

        backward_messages = []
        if not self.children:
            batch = self.backward_batches[0]
            emit = self.emit_mats[0]
            backward_msg = io_baum_welch.predict_inputs_marginal(
                self.trans_mat, emit, self.start_mat.T, batch, self.batch_lengths
            )
            backward_messages.append(backward_msg)
        else:
            for child, batch in zip(self.parent_sending_order, self.backward_batches):
                emit = self.dropout_emit(self.emit_mats[self.children.index(child)])
                backward_msg = io_baum_welch.predict_inputs_marginal(
                    self.trans_mat, emit, self.start_mat.T, batch, self.batch_lengths
                )
                backward_messages.append(backward_msg)

        accumulate_msg = self.accumulate_messages(backward_messages)
        for i, parent in enumerate(self.parents):
            parent.backward_batches.append(accumulate_msg)
            parent.parent_sending_order.append(self)

        self.forward_batches = []

    @staticmethod
    def accumulate_messages(messages):
        if len(messages) == 1:
            return messages[0]

        arr = np.stack(messages, axis=-1)
        combined_batch = mode(arr, axis=-1, keepdims=False).mode
        return combined_batch

    def dropout_emit(self, emit_mat):
        if self.dropout_rate == 0.0:
            return emit_mat

        mask = np.random.rand(*emit_mat.shape) > self.dropout_rate
        dropout_mat = emit_mat * mask
        normalize(dropout_mat, axis=2)
        return dropout_mat

    def evaluate(self):
        self.forward_batches = [self.accumulate_messages(self.forward_batches)]
        input_seqs = self.forward_batches[0]
        output_seqs = self.backward_batches[0]
        log_prob, _, _ = io_baum_welch.forward(
            self.trans_mat,
            self.emit_mats[0],
            self.start_mat,
            input_seqs,
            output_seqs,
            self.batch_lengths,
        )
        return log_prob

    def init_matrices(self, output_dimension=None):
        if self.is_initialized:
            return

        init = 1.0 / self.n_states
        self.trans_mat = self.random_state.dirichlet(np.full(self.n_states, init), size=(self.n_inputs, self.n_states))
        self.start_mat = self.random_state.dirichlet(np.full(self.n_states, init), size=self.n_inputs).T

        child_dimensions = [child.n_inputs for child in self.children] or [output_dimension]
        for dim in child_dimensions:
            emit_mat = self.random_state.rand(self.n_inputs, self.n_states, dim)
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