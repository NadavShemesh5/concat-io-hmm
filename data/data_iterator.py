import numpy as np


class BatchIterator:
    def __init__(self, npy_path, meta_path="data/io_processed_data/meta.npz", vocab_path="data/io_processed_data/vocab.npy", batch_size=None, seed=1):
        self.rng = np.random.default_rng(seed)

        # Load Data (Memory Mapped)
        self.dataset = np.load(npy_path, mmap_mode='r')

        # Defaults
        self.vocab_size = 0
        self.pad_idx = 0
        self.unk_idx = 1
        self.history_size = self.dataset.shape[1]

        # Load Metadata
        meta = np.load(meta_path)
        self.pad_idx = int(meta['pad_idx'])
        self.unk_idx = int(meta['unk_idx'])
        self.vocab_size = int(meta['vocab_size'])
        self.history_size = int(meta['history_size'])
        self.idx2token = np.load(vocab_path)

        self.num_rows = self.dataset.shape[0]
        self.batch_size = batch_size if batch_size is not None else self.num_rows

        self.indices = np.arange(self.num_rows)
        self.pos = 0
        self._reshuffle()

    def _reshuffle(self):
        self.indices = self.rng.permutation(self.num_rows)
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= self.num_rows:
            self._reshuffle()

        end = min(self.pos + self.batch_size, self.num_rows)
        batch_idx = self.indices[self.pos:end]
        self.pos = end

        batch = self.dataset[batch_idx, ...]
        lengths = np.sum(batch[:, 0, :] != self.pad_idx, axis=1)
        final_batch = self.pos >= self.num_rows
        return batch, lengths, final_batch
