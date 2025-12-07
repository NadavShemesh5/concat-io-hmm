import click
import numpy as np
from data.io_data_creation import load_dataset
from components.io_markov_node import IOMarkovNode


@click.command()
def main():
    # data = load_dataset()
    # train = data['train']['tokens']
    # valid = data['valid']['tokens']
    # vocab = data['vocab']
    # vocab_size = int(max(vocab['idx2token'].keys())) + 1

    train = [(np.array([0,1,2,3,4]), np.array([1,2,3,4,5])), (np.array([0,2,3]), np.array([2,3,5])), (np.array([0,1,3]), np.array([1,3,5]))]
    valid = [(np.array([0,1,2]), np.array([1,2,3])), (np.array([0,3,4]), np.array([3,4,5]))]
    vocab = {'idx2token': {i: str(i) for i in range(6)}, 'token2idx': {str(i): i for i in range(6)}}
    vocab_size = int(max(vocab['idx2token'].keys())) + 1

    model = IOMarkovNode(
        n_states=1,
        n_iter=100,
        n_inputs=vocab_size,
        n_outputs=vocab_size,
    )
    model.train(train, valid)

if __name__ == "__main__":
    main()