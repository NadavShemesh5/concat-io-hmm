import click

from data.data_creation import load_dataset
from components.markov_node import MarkovNode


@click.command()
@click.option("--node-transition", type=click.Choice(["deterministic", "wighted"]), default="deterministic")
@click.option("--partition-method", type=click.Choice(["information", "spectral"]), default="information")
def main(node_transition, partition_method):
    data = load_dataset()
    train = data['train']['tokens']
    valid = data['valid']['tokens']
    vocab = data['vocab']

    model = MarkovNode(
        n_states=32,
        n_iter=100,
    )
    model.train(train, vocab, valid=valid)


if __name__ == "__main__":
    main()