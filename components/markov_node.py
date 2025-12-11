import numpy as np
from tqdm import tqdm
from sklearn.utils.validation import check_random_state

from components.tools import normalize, timing
from algo import baum_welch


class MarkovNode:
    def __init__(
        self,
        n_states=1,
        start_prior=1e-6,
        trans_prior=1e-6,
        emit_prior=1e-6,
        random_state=1,
        n_iter=10,
    ):
        self.n_states = n_states
        self.start_prior = start_prior
        self.trans_prior = trans_prior
        self.emit_prior = emit_prior
        self.random_state = check_random_state(random_state)
        self.n_iter = n_iter
        self.is_trained = False

        self.n_tokens = None
        self.start_vec = None
        self.trans_mat = None
        self.emit_mat = None
        self.vocab = None

    def train(self, X, vocab, valid):
        self.init_parameters(vocab)
        for _ in range(self.n_iter):
            stats, curr_logprob = self.e_step(X)
            self.m_step(stats)

            perplexity = self.perplexity(X)
            print(f"Train Perplexity: {perplexity}")
            perplexity = self.perplexity(valid)
            print(f"Valid Perplexity: {perplexity}")

        self.is_trained = True
        return self

    @timing
    def e_step(self, X):
        stats = self.initialize_sufficient_statistics()
        curr_logprob = 0
        for sentence in tqdm(X):
            lattice, logprob, posteriors, fwdlattice, bwdlattice = self.fit(sentence)
            self.accumulate_sufficient_statistics(
                stats,
                sentence,
                lattice,
                posteriors,
                fwdlattice,
                bwdlattice,
            )
            curr_logprob += logprob
        return stats, curr_logprob

    def fit(self, sentence):
        frameprob = self.compute_likelihood(sentence)
        log_prob, fwdlattice, scaling_factors = baum_welch.forward(self.start_vec, self.trans_mat, frameprob)
        bwdlattice = baum_welch.backward(self.start_vec, self.trans_mat, frameprob, scaling_factors)
        posteriors = self.compute_posteriors(fwdlattice, bwdlattice)
        return frameprob, log_prob, posteriors, fwdlattice, bwdlattice

    def accumulate_sufficient_statistics(
        self, stats, sentence, lattice, posteriors, fwdlattice, bwdlattice
    ):
        stats["start"] += posteriors[0]

        xi_sum  = baum_welch.compute_xi_sum(
            fwdlattice,
            self.trans_mat,
            bwdlattice,
            lattice,
        )
        stats['trans'] += xi_sum

        np.add.at(stats["obs"].T, sentence, posteriors)

    @staticmethod
    def compute_posteriors(fwdlattice, bwdlattice):
        posteriors = fwdlattice * bwdlattice
        normalize(posteriors, axis=1)
        return posteriors

    def m_step(self, stats):
        self.start_vec = stats["start"] + self.start_prior
        normalize(self.start_vec)

        self.trans_mat = stats["trans"] + self.trans_prior
        normalize(self.trans_mat, axis=1)

        self.emit_mat = stats["obs"] + self.emit_prior
        normalize(self.emit_mat, axis=1)

    def initialize_sufficient_statistics(self):
        stats = {
            "start": np.zeros(self.n_states),
            "trans": np.zeros((self.n_states, self.n_states)),
            "obs": np.zeros((self.n_states, self.n_tokens)),
        }
        return stats

    def evaluate(self, X):
        log_prob = 0
        for sentence in X:
            frameprob = self.compute_likelihood(sentence)
            log_probij, _, _ = baum_welch.forward(self.start_vec, self.trans_mat, frameprob)
            log_prob += log_probij

        return log_prob

    def compute_likelihood(self, sentence):
        return self.emit_mat[:, sentence].T

    def perplexity(self, X):
        return np.exp(- self.evaluate(X) / sum(len(x) for x in X))

    def init_parameters(self, vocab):
        if self.is_trained:
            return

        init = 1.0 / self.n_states
        self.start_vec = self.random_state.dirichlet(np.full(self.n_states, init))

        self.trans_mat = self.random_state.dirichlet(np.full(self.n_states, init), size=self.n_states)

        self.vocab = vocab
        self.n_tokens = int(max(vocab['idx2token'].keys())) + 1
        self.emit_mat = self.random_state.rand(self.n_states, self.n_tokens)
        normalize(self.emit_mat, axis=1)
