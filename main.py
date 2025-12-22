import click

from components.markov_graph import MarkovGraph
from data.data_iterator import BatchIterator
from components.io_markov_node import IOMarkovNode


@click.command()
def main():
    # train_iter = BatchIterator("data/io_processed_data/train.npy", batch_size=4096)
    train_iter = BatchIterator("data/io_processed_data/valid.npy")
    valid_iter = BatchIterator("data/io_processed_data/valid.npy")

    graph = MarkovGraph(n_epochs=100)

    # input_node = IOMarkovNode(graph, n_states=32, input_index=0)
    # prev_layer = []
    # for i in range(6):
    #     node = IOMarkovNode(graph, n_states=32, n_inputs=64)
    #     input_node.add_child(node)
    #     prev_layer.append(node)
    # output_node = IOMarkovNode(graph, n_states=32, n_inputs=64, output_index=0)
    # for prev_node in prev_layer:
    #     prev_node.add_child(output_node)
    # input_node.add_child(output_node)

    # prev_layer = [IOMarkovNode(graph, n_states=32, input_index=0) for i in range(8)]
    # output_node = IOMarkovNode(graph, n_states=32, n_inputs=64, output_index=0)
    # for prev_node in prev_layer:
    #     prev_node.add_child(output_node)

    node_0 = IOMarkovNode(graph, n_states=1, input_index=0)
    node_1 = IOMarkovNode(graph, n_states=32, n_inputs=128)
    node_2 = IOMarkovNode(graph, n_states=32, n_inputs=128)
    node_3 = IOMarkovNode(graph, n_states=32, n_inputs=128)
    node_4 = IOMarkovNode(graph, n_states=32, n_inputs=128, output_index=0)
    node_0.add_child(node_1)
    node_0.add_child(node_2)
    node_0.add_child(node_3)
    node_0.add_child(node_4)
    node_1.add_child(node_2)
    node_1.add_child(node_3)
    node_1.add_child(node_4)
    node_2.add_child(node_3)
    node_2.add_child(node_4)
    node_3.add_child(node_4)

    # node_0 = IOMarkovNode(graph, n_states=32, input_index=0)
    # node_1 = IOMarkovNode(graph, n_states=32, n_inputs=32)
    # node_2 = IOMarkovNode(graph, n_states=32, n_inputs=32, output_index=0)
    # node_0.add_child(node_1)
    # node_1.add_child(node_2)

    graph.train(train_iter, valid_iter)


if __name__ == "__main__":
    main()