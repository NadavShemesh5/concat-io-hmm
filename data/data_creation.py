import os

import click
import numpy as np
import urllib.request
from collections import Counter


special_tokens = ["<unk>", "<eos>"]


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


def parse_paragraphs(lines):
    paragraphs = []
    curr_sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("="):
            curr_paragraph = " ".join(curr_sentences)
            if curr_paragraph:
                paragraphs.append(curr_paragraph)
            curr_sentences = []
            continue

        curr_sentences.append(line)

    curr_paragraph = " ".join(curr_sentences)
    if curr_paragraph:
        paragraphs.append(curr_paragraph)

    return paragraphs


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

    if partition == "paragraphs":
        return parse_paragraphs(lines)

    return parse_sentences(lines)


def tokenize(text, lower_case):
    text = text.lower() if lower_case else text
    tokens = text.split()
    return tokens


def encode_paragraphs(paragraphs, token2idx, lower_case):
    """
    Encode sentences to token indices

    Args:
        paragraphs: List of sentence strings
        token2idx: Dictionary mapping tokens to indices

    Returns:
        all_tokens: 1D numpy array of all tokens concatenated
        sentence_lengths: 1D numpy array of length of each sentence
    """
    all_tokens = []

    unk_idx = token2idx["<unk>"]
    eos_idx = token2idx["<eos>"]

    for paragraph in paragraphs:
        tokens = tokenize(paragraph, lower_case)
        indices = [token2idx.get(token, unk_idx) for token in tokens]
        indices.append(eos_idx)

        all_tokens.append(np.array(indices))

    return all_tokens


def build_vocab(texts, lower_case):
    """
    Build vocabulary from list of texts

    Args:
        texts: List of text strings
    """
    print("Building vocabulary...")

    token_counter = Counter()
    for text in texts:
        tokens = tokenize(text, lower_case)
        token_counter.update(tokens)

    token_counter.update(special_tokens)
    tokens = list(token_counter.keys())
    # random.shuffle(tokens)

    token2idx = {}
    # Add tokens to vocabulary
    for token in tokens:
        token2idx[token] = len(token2idx)

    # Create reverse mapping
    idx2token = {idx: token for token, idx in token2idx.items()}

    return token2idx, idx2token


def save_dataset(dataset, save_dir="./processed_data"):
    """Save processed dataset to disk"""
    os.makedirs(save_dir, exist_ok=True)

    # Save data splits
    np.save(os.path.join(save_dir, "train_tokens.npy"), np.array(dataset["train"]["tokens"], dtype=object), allow_pickle=True)
    np.save(os.path.join(save_dir, "valid_tokens.npy"), np.array(dataset["valid"]["tokens"], dtype=object), allow_pickle=True)
    np.save(os.path.join(save_dir, "test_tokens.npy"), np.array(dataset["test"]["tokens"], dtype=object), allow_pickle=True)

    # Save vocabulary
    np.save(os.path.join(save_dir, "token2idx.npy"), dataset["vocab"]["token2idx"], allow_pickle=True)
    np.save(os.path.join(save_dir, "idx2token.npy"), dataset["vocab"]["idx2token"], allow_pickle=True)

    print(f"\nDataset saved to {save_dir}/")


def load_dataset(save_dir="./data/processed_data"):
    """Load processed dataset from disk"""
    dataset = {
        "train": {
            "tokens": np.load(os.path.join(save_dir, "train_tokens.npy"), allow_pickle=True).tolist(),
        },
        "valid": {
            "tokens": np.load(os.path.join(save_dir, "valid_tokens.npy"), allow_pickle=True).tolist(),
        },
        "test": {
            "tokens": np.load(os.path.join(save_dir, "test_tokens.npy"), allow_pickle=True).tolist(),
        },
        "vocab": {
            "token2idx": np.load(
                os.path.join(save_dir, "token2idx.npy"), allow_pickle=True
            ).item(),
            "idx2token": np.load(
                os.path.join(save_dir, "idx2token.npy"), allow_pickle=True
            ).item(),
        },
    }

    return dataset


@click.command()
@click.option("--partition", type=click.Choice(["sentences", "paragraphs"]), default="sentences")
@click.option("-lc", "--lower-case", is_flag=True)
def main(partition, lower_case):
    extract_path = download_wikitext2()

    # Read files
    train_path = os.path.join(extract_path, "wiki.train.txt")
    valid_path = os.path.join(extract_path, "wiki.valid.txt")
    test_path = os.path.join(extract_path, "wiki.test.txt")

    print("Reading files...")
    train_paragraphs = read_file(train_path, partition)
    valid_paragraphs = read_file(valid_path, partition)
    test_paragraphs = read_file(test_path, partition)

    print(f"Train paragraphs: {len(train_paragraphs)}")
    print(f"Valid paragraphs: {len(valid_paragraphs)}")
    print(f"Test paragraphs: {len(test_paragraphs)}")

    # Build vocabulary from training data only
    token2idx, idx2token = build_vocab(train_paragraphs, lower_case)

    # Encode all splits
    print("\nEncoding train set...")
    train_tokens = encode_paragraphs(train_paragraphs, token2idx, lower_case)

    print("Encoding validation set...")
    valid_tokens = encode_paragraphs(valid_paragraphs, token2idx, lower_case)

    print("Encoding test set...")
    test_tokens = encode_paragraphs(test_paragraphs, token2idx, lower_case)

    print("\n=== Dataset Statistics ===")
    print(f"Vocabulary size: {len(token2idx):,}")

    save_dataset(
        {
            "train": {"tokens": train_tokens},
            "valid": {"tokens": valid_tokens},
            "test": {"tokens": test_tokens},
            "vocab": {"token2idx": token2idx, "idx2token": idx2token},
        }
    )


if __name__ == "__main__":
    main()