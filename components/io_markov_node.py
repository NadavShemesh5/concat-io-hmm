import numpy as np
from sklearn.utils.validation import check_random_state

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
            output_batch = self.backward_batches[idx]
            stats = self.init_emit_optimizer(stats, idx)
            self.fit_tensor_batch(input_batch, output_batch, stats, idx)
            self.optimize_emit(stats, idx)

        self.optimize_trans(stats)

    def init_trans_optimizer(self):
        stats = {"trans": np.zeros_like(self.trans_mat), "start": np.zeros_like(self.start_mat)}
        return stats

    def init_emit_optimizer(self, stats, idx):
        stats["emit"] = np.zeros_like(self.emit_mats[idx])
        return stats

    def optimize_trans(self, stats):
        stats["start"] += self.start_prior
        self.start_mat = stats["start"]
        normalize(self.start_mat, axis=1)

        stats["trans"] += self.trans_prior
        self.trans_mat = stats["trans"]
        normalize(self.trans_mat, axis=2)

    def optimize_emit(self, stats, idx):
        stats["emit"] += self.emit_prior
        self.emit_mats[idx] = stats["emit"]
        normalize(self.emit_mats[idx], axis=2)

    def fit_tensor_batch(self, input_seqs, output_seqs, stats, idx):
        trans_dynamic, emit_dynamic, start_dynamic = self.get_sentence_dynamics(input_seqs, output_seqs, idx)
        total_log_prob, fwd, scaling = io_baum_welch.forward(trans_dynamic, emit_dynamic, start_dynamic, self.batch_lengths)
        bwd = io_baum_welch.backward(trans_dynamic, emit_dynamic, scaling, self.batch_lengths)

        posteriors = self.compute_posteriors(fwd, bwd)
        io_baum_welch.compute_xi_sum(
            fwd, trans_dynamic, bwd, emit_dynamic, input_seqs, self.batch_lengths, stats["trans"]
        )

        start_tokens = input_seqs[:, 0]
        start_posteriors = posteriors[:, 0, :]
        np.add.at(stats["start"], start_tokens, start_posteriors)

        max_len = input_seqs.shape[1]
        mask = np.arange(max_len) < self.batch_lengths[:, None]

        valid_in = input_seqs[mask]
        valid_out = output_seqs[mask]
        valid_post = posteriors[mask]

        for k in range(self.n_states):
            np.add.at(
                stats["emit"][:, k, :],
                (valid_in, valid_out),
                valid_post[:, k]
            )

    def get_sentence_dynamics(self, input_seqs, output_seqs, idx):
        trans_dynamic = self.trans_mat[input_seqs, :]
        emit_dynamic = self.emit_mats[idx][input_seqs, :, output_seqs]
        start_dynamic = self.start_mat[input_seqs[:, 0], :]
        return trans_dynamic, emit_dynamic, start_dynamic

    @staticmethod
    def compute_posteriors(fwd, bwd):
        posteriors = fwd * bwd
        normalize(posteriors, axis=2)
        return posteriors

    @timing
    def send_forward_messages(self, choose_max=False):
        self.accumulate_forward()
        for child, emit in zip(self.children, self.emit_mats):
            forward_sample = self.sample_forward(self.forward_batches[0], emit, choose_max)
            child.forward_batches.append(forward_sample)

        if self.children:
            self.backward_batches = []

    @timing
    def send_backward_messages(self, choose_max=False):
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

        backward_sample = self.sample_backward(evidence_pairs, choose_max)
        for parent in self.parents:
            parent.backward_batches.append(backward_sample)
            parent.parent_sending_order.append(self)

        if self.parents:
            self.forward_batches = []

    def accumulate_forward(self):
        if len(self.forward_batches) == 1:
            return

        dims = tuple([self.n_inputs] * len(self.forward_batches)) if self.parents else self.n_inputs
        combined_batch = np.ravel_multi_index(self.forward_batches, dims)
        self.forward_batches = [combined_batch]

    def sample_forward(self, batch_data, emit_mat, choose_max):
        N, T = batch_data.shape
        sentence = batch_data
        output_seq = np.zeros((N, T), dtype=int)
        prev_state = np.zeros(N, dtype=int)

        for t in range(T):
            tokens = sentence[:, t]
            if t == 0:
                state_probs = self.start_mat[tokens, :]
            else:
                state_probs = self.trans_mat[tokens, prev_state, :]

            if choose_max:
                curr_state = np.argmax(state_probs, axis=1)
            else:
                curr_state = np.array([
                    self.random_state.choice(self.n_states, p=p)
                    for p in state_probs
                ])

            emit_probs = emit_mat[tokens, curr_state, :]
            if choose_max:
                output_token = np.argmax(emit_probs, axis=1)
            else:
                output_token = np.array([
                    self.random_state.choice(emit_mat.shape[-1], p=p)
                    for p in emit_probs
                ])

            output_seq[:, t] = output_token
            prev_state = curr_state

        return output_seq

    def sample_backward(self, output_pairs, choose_max):
        N, T = output_pairs[0][0].shape
        S = self.n_states

        input_seq = np.zeros((N, T), dtype=int)
        prev_state = np.zeros(N, dtype=int)
        trans_mat_T = self.trans_mat.transpose(1, 0, 2)

        for t in range(T):
            if t == 0:
                joint_probs = np.tile(self.start_mat[np.newaxis, :, :], (N, 1, 1)) # (N, U, S)
            else:
                joint_probs = trans_mat_T[prev_state]

            for (child_batch, child_emit) in output_pairs:
                y_t = child_batch[:, t]
                emit_slice = child_emit[:, :, y_t].transpose(2, 0, 1)
                joint_probs *= emit_slice

            flat_probs = joint_probs.reshape(N, -1)
            normalize(flat_probs, axis=1)
            if choose_max:
                indices = np.argmax(flat_probs, axis=1)
            else:
                indices = np.array([
                    self.random_state.choice(flat_probs.shape[1], p=p)
                    for p in flat_probs
                ])

            input_token = indices // S
            curr_state = indices % S

            input_seq[:, t] = input_token
            prev_state = curr_state

        return input_seq

    def evaluate(self):
        self.accumulate_forward()
        input_seqs = self.forward_batches[0]
        output_seqs = self.backward_batches[0]
        trans_dynamic, emit_dynamic, start_dynamic = self.get_sentence_dynamics(input_seqs, output_seqs, 0)
        total_log_prob, fwd, scaling = io_baum_welch.forward(trans_dynamic, emit_dynamic, start_dynamic, self.batch_lengths)
        return total_log_prob

    def init_matrices(self, output_dimension=None):
        if self.is_initialized:
            return

        input_dimension = int(np.prod([self.n_inputs for _ in range(len(self.parents))])) if self.parents else self.n_inputs
        init = 1.0 / self.n_states
        self.trans_mat = self.random_state.dirichlet(np.full(self.n_states, init), size=(input_dimension, self.n_states))
        self.start_mat = self.random_state.dirichlet(np.full(self.n_states, init), size=input_dimension)

        child_dimensions = [child.n_inputs for child in self.children] or [output_dimension]
        for dim in child_dimensions:
            emit_mat = self.random_state.rand(input_dimension, self.n_states, dim)
            normalize(emit_mat, axis=2)
            self.emit_mats.append(emit_mat)

        self.is_initialized = True

    def reset_batches(self):
        self.forward_batches = []
        self.backward_batches = []
        self.parent_sending_order = []
        self.batch_lengths = []

    def add_child(self, child):
        self.graph.add_edge(self, child)

    @property
    def children(self):
        return list(self.graph.successors(self))

    @property
    def parents(self):
        return list(self.graph.predecessors(self))
