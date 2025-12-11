class PseudoNode:
    def __init__(self, graph, data_iterator, n_inputs=0):
        self.graph = graph
        self.data_iterator = data_iterator
        self.n_inputs = n_inputs
        self.is_input_node = n_inputs == 0
        self.backward_batches = []
        self.forward_batches = []

    def fit_batch(self):
        pass

    def send_forward_messages(self, choose_max=False):
        if not self.is_input_node:
            return

        for child in self.children:
            child.forward_batches.append(self.data_iterator.current())

    def send_backward_messages(self, choose_max=False):
        if self.is_input_node:
            return

        for parent in self.parents:
            parent.backward_batches.append(self.data_iterator.current())

    def add_child(self, child):
        self.graph.add_edge(self, child)

    @property
    def children(self):
        return list(self.graph.successors(self))

    @property
    def parents(self):
        return list(self.graph.predecessors(self))
