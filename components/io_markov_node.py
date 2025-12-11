import numpy as np
from sklearn.utils.validation import check_random_state
from tqdm import tqdm
from scipy.special import softmax

from components.message import Message, Batch
from components.tools import normalize, timing
from algo import io_baum_welch


class IOMarkovNode:
    def __init__(
        self,
        graph,
        n_inputs,
        n_states=1,
        trans_prior=1e-6,
        emit_prior=1e-6,
        random_state=1,
    ):
        self.graph = graph
        self.n_inputs = n_inputs
        self.n_states = n_states
        self.trans_prior = trans_prior
        self.emit_prior = emit_prior
        self.random_state = check_random_state(random_state)

        init = 1.0 / self.n_states
        self.trans_mat = self.random_state.dirichlet(np.full(self.n_states, init), size=(self.n_inputs, self.n_states))
        self.emit_mats = []
        self.backward_batches = []
        self.forward_batches = []

    @timing
    def fit_batch(self):
        input_batch = self.accumulate_batches(self.forward_batches).messages
        stats = self.init_trans_optimizer()
        for idx, child in enumerate(self.children):
            stats = self.init_emit_optimizer(stats, child)
            output_batch = self.accumulate_batches(self.backward_batches).messages
            for input_seq, output_seq in tqdm(zip(input_batch, output_batch)):
                assert input_seq.content.size == output_seq.content.size
                self.fit_sentence(input_seq.content, output_seq.content, stats, idx)

            self.optimize_emit(stats, idx)

        self.optimize_trans(stats)

    def init_trans_optimizer(self):
        stats = {"trans": np.zeros((self.n_inputs, self.n_states, self.n_states))}
        return stats

    def init_emit_optimizer(self, stats, child):
        stats["obs"] = np.zeros((self.n_inputs, self.n_states, child.n_inputs))
        return stats

    @timing
    def optimize_emit(self, stats, idx):
        self.emit_mats[idx] = stats["obs"] + self.emit_prior
        normalize(self.emit_mats[idx], axis=2)

    def optimize_trans(self, stats):
        self.trans_mat = stats["trans"] + self.trans_prior
        normalize(self.trans_mat, axis=2)

    def fit_sentence(self, input_seq, output_seq, stats, idx):
        trans_dynamic = self.compute_transition_dynamics(input_seq)
        emit_dynamic = self.compute_emission_dynamic(input_seq, output_seq, idx)
        log_prob, fwd, scaling_factors = io_baum_welch.forward(trans_dynamic, emit_dynamic)
        bwd = io_baum_welch.backward(trans_dynamic, emit_dynamic, scaling_factors)
        posteriors = self.compute_posteriors(fwd, bwd)
        io_baum_welch.compute_xi_sum(
            fwd, trans_dynamic, bwd, emit_dynamic, input_seq, stats["trans"]
        )

        for k in range(self.n_states):
            np.add.at(stats["obs"][:, k, :], (input_seq, output_seq), posteriors[:, k])

    @staticmethod
    def compute_posteriors(fwd, bwd):
        posteriors = fwd * bwd
        normalize(posteriors, axis=1)
        return posteriors

    def compute_transition_dynamics(self, input_seq):
        return self.trans_mat[input_seq]

    def compute_emission_dynamic(self, input_seq, output_seq, idx):
        return self.emit_mats[idx][input_seq, :, output_seq]

    @timing
    def send_forward_messages(self, choose_max=False):
        batch = self.accumulate_batches(self.forward_batches)
        for child, emit in zip(self.children, self.emit_mats):
            forward_batch = Batch([self.sample_forward(message, emit, choose_max) for message in batch.messages])
            child.forward_batches.append(forward_batch)

        self.backward_batches = []

    @timing
    def send_backward_messages(self, choose_max=False):
        batch = self.accumulate_batches(self.backward_batches)
        for parent in self.parents:
            emit = self.emit_mats[self.parents.index(parent)]
            backward_batch = Batch([self.sample_backward(message, emit, choose_max) for message in batch.messages])
            parent.backward_batches.append(backward_batch)

        self.forward_batches = []

    @staticmethod
    @timing
    def accumulate_batches(batches):
        if len(batches) == 1:
            return batches[0]

        n_messages = len(batches[0].messages)
        messages = []
        for i in range(n_messages):
            probs = softmax(np.array([batch.messages[i].prob for batch in batches]))
            # TODO: check if softmax is the appropriate choice here
            idx = int(np.random.choice(probs.size, p=probs))
            curr_messages = [batch.messages[i] for batch in batches]
            messages.append(curr_messages[idx])

        return Batch(messages)

    def sample_forward(self, message, emit_mat, choose_max):
        sentence = message.content
        output_seq = np.zeros(len(sentence), dtype=int)
        log_prob = message.prob

        n_outputs = emit_mat.shape[2]
        prev_state = self.random_state.choice(self.n_states)
        for t, token in enumerate(sentence):
            trans_probs = self.trans_mat[token, prev_state, :]
            curr_state = np.argmax(trans_probs) if choose_max else self.random_state.choice(self.n_states, p=trans_probs)
            log_prob += np.log(trans_probs[curr_state] + 1e-15)

            emit_probs = emit_mat[token, curr_state, :]
            output_token = np.argmax(trans_probs) if choose_max else self.random_state.choice(n_outputs, p=emit_probs)
            output_seq[t] = output_token

            log_prob += np.log(emit_probs[output_token] + 1e-15)
            prev_state = curr_state

        return Message(output_seq, log_prob)

    def sample_backward(self, message, emit_mat, choose_max):
        output_seq = message.content
        input_seq = np.zeros(len(output_seq), dtype=int)

        log_prob = message.prob
        prev_state = self.random_state.choice(self.n_states)
        for t, output_token in enumerate(output_seq):
            trans_slice = self.trans_mat[:, prev_state, :]
            emit_slice = emit_mat[:, :, output_token]

            joint_probs = trans_slice * emit_slice
            flat_probs = joint_probs.flatten()
            sum_prob = flat_probs.sum()
            flat_probs /= sum_prob

            idx =  np.argmax(flat_probs) if choose_max else self.random_state.choice(flat_probs.size, p=flat_probs)
            input_token = idx // self.n_states
            curr_state = idx % self.n_states
            input_seq[t] = input_token

            log_prob += np.log(joint_probs[input_token, curr_state] + 1e-15)
            prev_state = curr_state

        return Message(input_seq, log_prob)

    def evaluate(self):
        log_prob = 0
        input_batch = self.accumulate_batches(self.forward_batches).messages
        for idx, _ in enumerate(self.children):
            output_batch = self.accumulate_batches(self.backward_batches).messages
            for input_seq, output_seq in tqdm(zip(input_batch, output_batch)):
                trans_dynamic = self.compute_transition_dynamics(input_seq.content)
                emit_dynamic = self.compute_emission_dynamic(input_seq.content, output_seq.content, idx)
                l_prob, _, _ = io_baum_welch.forward(trans_dynamic, emit_dynamic)
                log_prob += l_prob

        return log_prob

    def add_child(self, child):
        self.graph.add_edge(self, child)
        emit_mat = self.random_state.rand(self.n_inputs, self.n_states, child.n_inputs)
        normalize(emit_mat, axis=2)
        self.emit_mats.append(emit_mat)

    @property
    def children(self):
        return list(self.graph.successors(self))

    @property
    def parents(self):
        return list(self.graph.predecessors(self))
