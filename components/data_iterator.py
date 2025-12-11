import random

from components.message import Message, Batch


class BatchIterator:
    def __init__(self, pairs, batch_size=128, shuffle=True):
        self.data = [[entry[ind] for ind in range(len(entry))] for entry in pairs]
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.order = list(range(len(pairs)))
        self.pos = 0
        self.current_indices = None
        self.new_epoch()
        self.current_batch = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.order):
            self.new_epoch()

        start = self.pos
        end = start + self.batch_size
        self.current_indices = self.order[start:end]
        self.pos = end

        curr_batch  = [self.data[i] for i in self.current_indices]
        self.current_batch = curr_batch

    def new_epoch(self):
        self.pos = 0
        self.current_indices = None
        if self.shuffle:
            random.shuffle(self.order)

class View:
    def __init__(self, parent, idx):
        self.idx = idx
        self.parent = parent

    def current(self):
        return Batch([Message(entry[self.idx], 0) for entry in self.parent.current_batch for _ in range(1)])
