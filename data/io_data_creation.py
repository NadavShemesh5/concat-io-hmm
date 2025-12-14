import os

import click
import numpy as np
import urllib.request
from collections import Counter


special_tokens = ["<pad>", "<sos>", "<unk>", "<eos>"]


def download_wikitext2(data_dir="./raw_data"):
    """Download WikiText-2 dataset if not already present"""
    os.makedirs(data_dir, exist_ok=True)

    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    valid_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
    test_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"

    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "wiki.train.txt")
    valid_path = os.path.join(data_dir, "wiki.valid.txt")
    test_path = os.path.join(data_dir, "wiki.test.txt")

    if not os.path.exists(train_path):
        print("Downloading WikiText-2 train set...")
        urllib.request.urlretrieve(url, train_path)

        print("Downloading WikiText-2 valid set...")
        urllib.request.urlretrieve(valid_url, valid_path)

        print("Downloading WikiText-2 test set...")
        urllib.request.urlretrieve(test_url, test_path)

        print("Download complete!")
    else:
        print("WikiText-2 already downloaded.")

    return data_dir


def parse_sentences(lines):
    sentences = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("="):
            sentences.append(line)

    return sentences


def read_file(filepath, partition):
    """Read and preprocess WikiText file"""
    with open(filepath, "r") as f:
        lines = f.readlines()

    return parse_sentences(lines)


def tokenize(text, lower_case):
    text = text.lower() if lower_case else text
    tokens = text.split()
    return tokens


def encode_sentences(sentences, token2idx, lower_case, history_size):
    unk_idx = token2idx["<unk>"]
    eos_idx = token2idx["<eos>"]
    sos_idx = token2idx["<sos>"]
    pad_idx = token2idx["<pad>"]

    encoded_list = []
    max_width = 0
    for sentence in sentences:
        tokens = tokenize(sentence, lower_case)
        indices = [token2idx.get(token, unk_idx) for token in tokens]
        max_width = max(max_width, len(indices) + 1)  # +1 for eos

        history = [indices + [eos_idx]]
        for i in range(history_size):
            curr_in_indices = [pad_idx] * i + [sos_idx] + indices[: len(indices) - i]
            history.append(curr_in_indices)

        encoded_list.append(history)

    data = np.full((len(encoded_list), history_size + 1, max_width), pad_idx, dtype=np.uint16)

    # Fill Data
    for i, seq in enumerate(encoded_list):
        for j, sentence in enumerate(seq):
            l = min(len(sentence), max_width)
            data[i, j, :l] = sentence[:l]

    return data


def build_vocab(sentences, lower_case):
    counter = Counter()
    for p in sentences:
        if lower_case: p = p.lower()
        counter.update(p.split())

    # Sort by frequency
    vocab = special_tokens + [word for word, _ in counter.most_common()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = np.array(vocab)
    return token2idx, idx2token


@click.command()
@click.option('--partition', default=1.0, help='Fraction of data to use')
@click.option('--lower_case', is_flag=True, default=False)
@click.option('--history_size', default=8, type=int, help='Content sequence length')
def main(partition, lower_case, history_size):
    extract_path = download_wikitext2()

    train_path = os.path.join(extract_path, "wiki.train.txt")
    valid_path = os.path.join(extract_path, "wiki.valid.txt")
    test_path = os.path.join(extract_path, "wiki.test.txt")

    print("Reading files...")
    train_sentences = read_file(train_path, partition)
    valid_sentences = read_file(valid_path, partition)
    test_sentences = read_file(test_path, partition)

    print(f"Train paragraphs: {len(train_sentences)}")
    print(f"Valid paragraphs: {len(valid_sentences)}")
    print(f"Test paragraphs: {len(test_sentences)}")

    token2idx, idx2token = build_vocab(train_sentences, lower_case)

    print("\nEncoding Train...")
    train_np = encode_sentences(train_sentences, token2idx, lower_case, history_size)

    print("Encoding Validation...")
    valid_np = encode_sentences(valid_sentences, token2idx, lower_case, history_size)

    print("Encoding Test...")
    test_np = encode_sentences(test_sentences, token2idx, lower_case, history_size)

    output_dir = "./io_processed_data"
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "train.npy"), train_np)
    np.save(os.path.join(output_dir, "valid.npy"), valid_np)
    np.save(os.path.join(output_dir, "test.npy"), test_np)

    np.save(os.path.join(output_dir, "vocab.npy"), idx2token)
    np.savez(os.path.join(output_dir, "meta.npz"),
             pad_idx=token2idx["<pad>"],
             unk_idx=token2idx["<unk>"],
             sos_idx=token2idx["<sos>"],
             eos_idx=token2idx["<eos>"],
             vocab_size=len(idx2token),
             history_size=history_size)

    print(f"\nDone. Saved to {output_dir}")
    print(f"Vocab Size: {len(idx2token)}")
    print(f"Pad Index: {token2idx['<pad>']}")

if __name__ == "__main__":
    main()