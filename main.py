import click

from components.markov_graph import MarkovGraph
from data.data_iterator import BatchIterator
from components.io_markov_node import IOMarkovNode


@click.command()
def main():
    train_iter = BatchIterator("data/io_processed_data/train.npy", batch_size=32)
    valid_iter = BatchIterator("data/io_processed_data/valid.npy")

    graph = MarkovGraph(n_iter=100)

    input_0 = IOMarkovNode(graph, n_states=1, input_index=0)
    input_1 = IOMarkovNode(graph, n_states=2, input_index=1)
    input_2 = IOMarkovNode(graph, n_states=2, input_index=2)
    input_3 = IOMarkovNode(graph, n_states=2, input_index=3)
    hidden_0 = IOMarkovNode(graph, n_states=32, n_inputs=4)
    output_0 = IOMarkovNode(graph, n_states=1, n_inputs=32, output_index=0)
    input_0.add_child(hidden_0)
    input_1.add_child(hidden_0)
    input_2.add_child(hidden_0)
    input_3.add_child(hidden_0)
    hidden_0.add_child(output_0)

    # node_0 = IOMarkovNode(graph, n_states=32, input_index=0)
    # node_1 = IOMarkovNode(graph, n_states=32, n_inputs=512, output_index=0)
    # node_0.add_child(node_1)

    # node_0 = IOMarkovNode(graph, n_states=1, input_index=0, output_index=0)
    # graph.add_node(node_0)

    graph.train(train_iter, valid_iter)

if __name__ == "__main__":
    main()