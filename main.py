import click

from data.io_data_creation import load_dataset
from components.io_markov_node import IOMarkovNode


@click.command()
def main():
    data = load_dataset()
    train = data['train']['tokens']
    valid = data['valid']['tokens']
    vocab = data['vocab']
    vocab_size = int(max(vocab['idx2token'].keys())) + 1

    # train = [([0,1,2,3,4], [1,2,3,4,5]), ([0,2,3], [2,3,5]), ([0,1,3], [1,3,5])]
    # valid = [([0,1,2], [1,2,3]), ([0,3,4], [3,4,5])]
    # vocab = {'idx2token': {i: str(i) for i in range(6)}, 'token2idx': {str(i): i for i in range(6)}}
    # vocab_size = int(max(vocab['idx2token'].keys())) + 1

    model = IOMarkovNode(
        n_states=2,
        n_iter=100,
        n_inputs=vocab_size,
        n_outputs=vocab_size,
    )
    model.train(train, vocab, valid=valid)
    print(model.trans_mat)

if __name__ == "__main__":
    main()