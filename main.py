import click

from components.data_iterator import BatchIterator, View
from components.markov_graph import MarkovGraph
from components.pseudo_node import PseudoNode
from data.io_data_creation import load_dataset
from components.io_markov_node import IOMarkovNode


@click.command()
def main():
    data = load_dataset()
    train = data['train']['tokens']
    valid = data['valid']['tokens']
    vocab = data['vocab']
    vocab_size = int(max(vocab['idx2token'].keys())) + 1

    data_coordinator = BatchIterator(train, batch_size=256)
    # data_coordinator = BatchIterator(train, batch_size=len(train))
    input_data = View(data_coordinator, 0)
    output_data = View(data_coordinator, 1)

    graph = MarkovGraph(n_iter=10)
    data_input = PseudoNode(graph, input_data)
    node_1 = IOMarkovNode(graph, n_inputs=vocab_size, n_states=32)
    node_2 = IOMarkovNode(graph, n_inputs=32, n_states=32)
    node_3 = IOMarkovNode(graph, n_inputs=32, n_states=32)
    node_4 = IOMarkovNode(graph, n_inputs=32, n_states=32)
    node_5 = IOMarkovNode(graph, n_inputs=32, n_states=32)
    node_6 = IOMarkovNode(graph, n_inputs=32, n_states=32)
    node_7 = IOMarkovNode(graph, n_inputs=32, n_states=32)
    data_output = PseudoNode(graph, output_data, n_inputs=vocab_size)

    data_input.add_child(node_1)
    node_1.add_child(node_2)
    node_2.add_child(node_3)
    node_3.add_child(node_4)
    node_4.add_child(node_5)
    node_5.add_child(node_6)
    node_6.add_child(node_7)
    node_7.add_child(data_output)

    # graph = MarkovGraph(n_iter=10)
    # data_input = PseudoNode(graph, input_data)
    # node_1 = IOMarkovNode(graph, n_inputs=vocab_size, n_states=1)
    # data_output = PseudoNode(graph, output_data, n_inputs=vocab_size)
    #
    # data_input.add_child(node_1)
    # node_1.add_child(data_output)

    graph.train(data_coordinator)

if __name__ == "__main__":
    main()