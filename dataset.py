import logging
import argparse
import pickle
import re

import torch
import nltk
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Small_Transformers_Dataset(Dataset):
    def __init__(self, index_stories, vocab, idx2word):
        self.index_stories = index_stories
        self.vocab = vocab
        self.idx2word = idx2word

    def __len__(self):
        return len(self.index_stories)

    def __getitem__(self, idx):
        story = self.index_stories[idx]
        story_shifted = story[1:] + [self.vocab["<pad>"]]

        return torch.tensor(story), torch.tensor(story_shifted)


def create_dataloader_from_file(
    dataset,
    max_seq_length,
    fraction_wanted,
    train_batch_size,
    val_batch_size,
    max_vocab_size,
    vocab_file=None,
):
    hf_dataset = load_dataset(dataset)

    train_data = hf_dataset["train"]
    val_data = hf_dataset["validation"]

    train_stories = []
    val_stories = []

    word_count = {}

    for i in tqdm(
        range(int(fraction_wanted * len(train_data))), desc="Processing train data"
    ):
        story = train_data[i]["text"]
        story = story.lower()
        story = re.sub(r"[^a-zA-Z\s]", " ", story)
        story = re.sub(r"\s+", " ", story)

        story = story.strip()
        story = nltk.word_tokenize(story)
        # story = ["<bos>"] + story + ["<eos>"]

        if vocab_file is None:
            for word in story:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

        train_stories.append(story)

    for i in tqdm(
        range(int(fraction_wanted * len(val_data))), desc="Processing val data"
    ):
        story = val_data[i]["text"]
        story = story.lower()
        story = re.sub(r"[^a-zA-Z\s]", " ", story)
        story = re.sub(r"\s+", " ", story)

        story = story.strip()
        story = nltk.word_tokenize(story)
        # story = ["<bos>"] + story + ["<eos>"]

        if vocab_file is None:
            for word in story:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

        val_stories.append(story)

    if vocab_file is not None:
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)
            assert len(vocab) == max_vocab_size

    else:
        # sort word_count in descending order
        word_count = dict(
            sorted(word_count.items(), key=lambda item: item[1], reverse=True)
        )

        # select the top max_vocab_size words
        top_words = list(word_count.keys())[: max_vocab_size - 4]
        top_words = ["<bos>", "<eos>", "<pad>", "<unk>"] + top_words
        vocab = {k: v for v, k in enumerate(top_words)}

        # print(vocab)

    # else:
    #     vocab = {}
    #     vocab["<bos>"] = len(vocab)
    #     vocab["<eos>"] = len(vocab)
    #     vocab["<pad>"] = len(vocab)
    #     vocab["<unk>"] = len(vocab)

    #     for data in [train_stories, val_stories]:
    #         for story in data:
    #             for word in story:
    #                 if word_count[word] >= vocab_threshold and word not in vocab.keys():
    #                     vocab[word] = len(vocab)

    #     with open("vocab.pkl", "wb") as f:
    #         pickle.dump(vocab, f)

    idx2word = {idx: word for word, idx in vocab.items()}

    index_train_stories = []
    padded_train_stories = []
    index_val_stories = []
    padded_val_stories = []

    for story in tqdm(train_stories, desc="Encoding train stories"):
        index_story = [vocab.get(word, vocab["<unk>"]) for word in story]
        new_story = story[: max_seq_length - 2]
        # add bos eos
        new_story = ["<bos>"] + new_story + ["<eos>"]
        new_story = new_story + ["<pad>"] * (max_seq_length - len(new_story))

        index_story = index_story[: max_seq_length - 2]
        index_story = [vocab["<bos>"]] + index_story + [vocab["<eos>"]]
        index_story = index_story + [vocab["<pad>"]] * (
            max_seq_length - len(index_story)
        )

        index_train_stories.append(index_story)
        padded_train_stories.append(new_story)

    for story in tqdm(val_stories, desc="Encoding val stories"):
        index_story = [vocab.get(word, vocab["<unk>"]) for word in story]
        new_story = story[: max_seq_length - 2]
        # add bos eos
        new_story = ["<bos>"] + new_story + ["<eos>"]
        new_story = new_story + ["<pad>"] * (max_seq_length - len(new_story))

        index_story = index_story[: max_seq_length - 2]
        index_story = [vocab["<bos>"]] + index_story + [vocab["<eos>"]]
        index_story = index_story + [vocab["<pad>"]] * (
            max_seq_length - len(index_story)
        )

        index_val_stories.append(index_story)
        padded_val_stories.append(new_story)

    train_dataset = Small_Transformers_Dataset(index_train_stories, vocab, idx2word)
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=False
    )

    val_dataset = Small_Transformers_Dataset(index_val_stories, vocab, idx2word)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def main():

#     def create_dataloader_from_file(
#     dataset,
#     max_seq_length,
#     fraction_wanted,
#     train_batch_size,
#     val_batch_size,
#     max_vocab_size,
#     vocab_file=None,
# ):

    train_dataloader, val_dataloader = create_dataloader_from_file(
        "roneneldan/TinyStories", 512, 0.0005, 64, 8, 1500,
    )

    for batch_idx, (x, y) in enumerate(train_dataloader):
        print(x)
        print(y)
        break
    for batch_idx, (x, y) in enumerate(val_dataloader):
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
        default="artifacts/vocab.json",
    )

    # Parse the arguments
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)

    main()
