import click

from components.markov_graph import MarkovGraph
from data.data_iterator import BatchIterator
from components.io_markov_node import IOMarkovNode


@click.command()
def main():
    train_iter = BatchIterator("data/io_processed_data/train.npy", batch_size=64)
    # train_iter = BatchIterator("data/io_processed_data/valid.npy")
    valid_iter = BatchIterator("data/io_processed_data/valid.npy")

    graph = MarkovGraph(n_epochs=100)

    # input_node = IOMarkovNode(graph, n_states=64, input_index=0)
    # output_node = IOMarkovNode(graph, n_states=64, n_inputs=128, output_index=0)
    # existing = [input_node]
    # for i in range(1, 8):
    #     for j in range(i):
    #         new_node = IOMarkovNode(graph, n_states=2, n_inputs=2)
    #         for node in existing:
    #             node.add_child(new_node)
    #         existing.append(new_node)
    #         new_node.add_child(output_node)

    # node_0 = IOMarkovNode(graph, n_states=64, input_index=0)
    # node_1 = IOMarkovNode(graph, n_states=64, n_inputs=64)
    # node_2 = IOMarkovNode(graph, n_states=64, n_inputs=64)
    # node_3 = IOMarkovNode(graph, n_states=64, n_inputs=64)
    # node_4 = IOMarkovNode(graph, n_states=64, n_inputs=64, output_index=0)
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

    # output_node = IOMarkovNode(graph, n_states=64, n_inputs=64, dropout_rate=0.0, output_index=0)
    # for i in range(7):
    #     temp_node = IOMarkovNode(graph, n_states=(i+1), input_index=0, dropout_rate=0.0)
    #     temp_node.add_child(output_node)

    output_node = IOMarkovNode(graph, n_states=8, n_inputs=8, output_index=0)
    input_node = IOMarkovNode(graph, n_states=8, input_index=0, dropout_rate=0.0)
    input_node.add_child(output_node)

    graph.train(train_iter, valid_iter)


if __name__ == "__main__":
    main()