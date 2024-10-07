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


def create_dataloder_from_file(
    dataset,
    max_seq_length,
    fraction_wanted,
    vocab_threshold,
    train_batch_size,
    val_batch_size,
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
        story = ["<bos>"] + story + ["<eos>"]

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
            story = ["<bos>"] + story + ["<eos>"]

            for word in story:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

            val_stories.append(story)

        if vocab_file is not None:
            with open(vocab_file, "rb") as f:
                vocab = pickle.load(f)
        else:
            vocab = {}
            vocab["<bos>"] = len(vocab)
            vocab["<eos>"] = len(vocab)
            vocab["<pad>"] = len(vocab)
            vocab["<unk>"] = len(vocab)

            for data in [train_stories, val_stories]:
                for story in data:
                    for word in story:
                        if (
                            word_count[word] >= vocab_threshold
                            and word not in vocab.keys()
                        ):
                            vocab[word] = len(vocab)

            with open("vocab.pkl", "wb") as f:
                pickle.dump(vocab, f)

        idx2word = {idx: word for word, idx in vocab.items()}

        index_train_stories = []
        padded_train_stories = []
        index_val_stories = []
        padded_val_stories = []

        for story in tqdm(train_stories, desc="Encoding train stories"):
            index_story = [vocab.get(word, vocab["<unk>"]) for word in story]
            new_story = story[:max_seq_length]
            new_story = new_story + ["<pad>"] * (max_seq_length - len(new_story))

            if len(index_story) > max_seq_length:
                index_story = index_story[:max_seq_length]
            else:
                index_story = index_story + [vocab["<pad>"]] * (
                    max_seq_length - len(index_story)
                )

            index_train_stories.append(index_story)
            padded_train_stories.append(new_story)

        for story in tqdm(val_stories, desc="Encoding val stories"):
            index_story = [vocab.get(word, vocab["<unk>"]) for word in story]
            new_story = story[:max_seq_length]
            new_story = new_story + ["<pad>"] * (max_seq_length - len(new_story))

            if len(index_story) > max_seq_length:
                index_story = index_story[:max_seq_length]
            else:
                index_story = index_story + [vocab["<pad>"]] * (
                    max_seq_length - len(index_story)
                )

            index_val_stories.append(index_story)
            padded_val_stories.append(new_story)

        train_dataset = Small_Transformers_Dataset(index_train_stories, vocab, idx2word)
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )

        val_dataset = Small_Transformers_Dataset(index_val_stories, vocab, idx2word)
        val_dataloader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False
        )

        return train_dataloader, val_dataloader


if __name__ == "__main__":
    train_dataloader, val_dataloader = create_dataloder_from_file(
        "roneneldan/TinyStories", 512, 0.0005, 5, 64, 8
    )

    for batch_idx, (x, y) in enumerate(train_dataloader):
        print(x)
        print(y)
        break
    for batch_idx, (x, y) in enumerate(val_dataloader):
        print(x)
        print(y)
        break
