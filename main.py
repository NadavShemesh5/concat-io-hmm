import click

from components.markov_graph import MarkovGraph
from data.data_iterator import BatchIterator
from components.io_markov_node import IOMarkovNode


@click.command()
def main():
    train_iter = BatchIterator("data/io_processed_data/train.npy", batch_size=256)
    valid_iter = BatchIterator("data/io_processed_data/valid.npy")

    graph = MarkovGraph(n_iter=100)

    # prev_layer = [IOMarkovNode(graph, n_states=1, input_index=0)]
    # for l in range(8):
    #     layer = []
    #     for i in range(8):
    #         node = IOMarkovNode(graph, n_states=1, n_inputs=2)
    #         for prev_node in prev_layer:
    #             prev_node.add_child(node)
    #
    #         layer.append(node)
    #
    #     prev_layer = layer
    #
    # output_node = IOMarkovNode(graph, n_states=1, n_inputs=2, output_index=0)
    # for prev_node in prev_layer:
    #     prev_node.add_child(output_node)

    # node_0 = IOMarkovNode(graph, n_states=1, input_index=0)
    # node_1 = IOMarkovNode(graph, n_states=32, n_inputs=32)
    # node_2 = IOMarkovNode(graph, n_states=16, n_inputs=16)
    # node_3 = IOMarkovNode(graph, n_states=8, n_inputs=8)
    # node_4 = IOMarkovNode(graph, n_states=4, n_inputs=4, output_index=0)
    # node_0.add_child(node_1)
    # node_0.add_child(node_2)
    # node_0.add_child(node_3)
    # node_0.add_child(node_4)
    # node_1.add_child(node_2)
    # node_1.add_child(node_3)
    # node_1.add_child(node_4)
    # node_2.add_child(node_3)
    # node_2.add_child(node_4)
    # node_3.add_child(node_4)

    node_0 = IOMarkovNode(graph, n_states=8, input_index=0)
    node_1 = IOMarkovNode(graph, n_states=8, n_inputs=64, output_index=0)
    node_0.add_child(node_1)

    # node_0 = IOMarkovNode(graph, n_states=1, input_index=0, output_index=0)
    # graph.add_node(node_0)

    graph.train(train_iter, valid_iter)


if __name__ == "__main__":
    main()