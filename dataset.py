import logging
import argparse
import re
import os

import torch
import datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import tokenizers


class Small_Transformers_Dataset(Dataset):
    def __init__(self, stories):
        self.stories = stories

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]["ids"]
        story_shifted = story[1:] + [0]

        return torch.tensor(story), torch.tensor(story_shifted)


# Creates a batch iterator from a dataset so that it is easier to train
# a tokenizer from a dataset. Instead of writing our own functions that
# may or may not be fast, we leverage HF's dataset functions to select
# relevant columns and iterate through the dataset quickly
def batch_iterator(dataset, batch_size=64):
    for batch in dataset.select_columns("text").iter(batch_size=batch_size):
        yield batch["text"]


# Tokenize data. This ensures that each sentence is tokenized correctly
# to be stored and loaded in the dataloader
def tokenize_batch(batch, tknizer):
    encoded_sentences = tknizer.encode_batch_fast(batch['text'])
    return {
        "text": batch['text'],
        # TODO: convert all these to tensors
        "ids": [item.ids for item in encoded_sentences]
    }


def create_dataloader_from_file(
    dataset,
    max_seq_length,
    fraction_wanted,
    train_batch_size,
    val_batch_size,
    max_vocab_size,
    vocab_file=None,
):
    logging.info("Loading dataset")
    hf_dataset = datasets.load_dataset(dataset)

    # Filter data to a manageable size to prevent things from going out of hand
    logging.info("Filtering dataset to smaller size based on wanted fraction")
    inv_fraction = 1 / fraction_wanted
    hf_dataset = hf_dataset.filter(lambda _, idx: idx % inv_fraction == 0, with_indices=True)

    # Remove samples with very short or very long sentences. Emperically, this
    # represents a small, but very poor sample of the dataset.
    hf_dataset = hf_dataset.filter(lambda e: len(e['text']) > 100 and len(e['text']) < 4096)


    tokenizer_model = None
    # Setup HF Tokenizer
    if vocab_file is not None and os.path.exists(vocab_file):
        logging.info("Loading existing tokenizer...")
        tokenizer_model = tokenizers.Tokenizer.from_file(vocab_file)
        # TODO: Make sure that the size of the dataset kept + vocab size is checked while loading the dataset
    
    if tokenizer_model is None:
        logging.info("Vocab file either not provided or not found. Training tokenizer...")
        tokenizer_model = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
        tokenizer_model.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

        from tokenizers.normalizers import (
            Sequence,
            Replace,
            Strip,
            Lowercase,
        )
        tokenizer_model.normalizer = Sequence([
            # Basic cleanup
            Lowercase(),
            Strip(),                                   # Remove leading/trailing whitespace
            Replace(r"\\", ""),                        # Remove backslashes
            Replace(r"  +", " "),                      # Replace multiple spaces with single

            # Standardize dashes and quotes
            Replace(r"–", "-"),                        # Standardize dashes
            Replace(r" — ", " - "),
            Replace(r"—", " - "),
            Replace(r"…", "..."),                      # Standardize ellipsis
            Replace(r""", '"'),                        # Standardize quotes
            Replace(r""", '"'),
            Replace(r"'", "'"),                        # Standardize apostrophes

            # Remove unwanted special characters
            Replace(
                r".*[|</*`=_&@~#%\[\]+()].*", "",
            ),

            # Remove texts with invalid characters (ord > 127 or < 32, except newline)
            Replace(
                r".*[^\x0A\x20-\x7F].*", "",
            ),
        ])

        tokenizer_model.post_tokenizer = tokenizers.processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )

        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=max_vocab_size,
            special_tokens=['[PAD]', '[BOS]', '[EOS]', '[UNK]'],
            show_progress=True,
        )

        tokenizer_model.train_from_iterator(
            batch_iterator(hf_dataset['train'], batch_size=256),
            trainer=trainer
        )

        tokenizer_model.enable_truncation(
            max_length=max_seq_length
        )

        tokenizer_model.enable_padding(
            direction='right',
            pad_token='[PAD]',
            pad_id=0,
            length=max_seq_length
        )

    logging.info("Tokenizing sequences...")
    hf_dataset = hf_dataset.map(tokenize_batch, batched=True, fn_kwargs={'tknizer': tokenizer_model})

    logging.info("Returning output")

    train_dataset = Small_Transformers_Dataset(hf_dataset['train'])
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True
    )

    val_dataset = Small_Transformers_Dataset(hf_dataset['validation'])
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def main():
    train_dataloader, val_dataloader = create_dataloader_from_file(
        "roneneldan/TinyStories", 512, 0.05, 64, 8, 1500,
    )

    for x, y in train_dataloader:
        print(x)
        print(y)
        break
    for x, y in val_dataloader:
        print(x)
        print(y)
        break


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Setup the dataset")

    # Allow for logging support throughout the app
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    parser.add_argument(
        "-s",
        "--seed",
        help="Force seed for deterministic randomness",
        action="store",
        dest="seed",
    )

    parser.add_argument(
        "--vocab-output",
        help="Location of vocabulary file",
        action="store",
        dest="vocab_output",
        default="artifacts/vocab.pkl",
    )

    # Parse the arguments
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)

    main()
