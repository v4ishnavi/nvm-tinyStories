import re
from collections import Counter
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Load the TinyStories dataset
ds = load_dataset("roneneldan/TinyStories")
print("Structure of the dataset:")
print(ds)
# Create a reduced vocabulary
def create_reduced_vocab_and_save(ds, vocab_size_limit=1500, output_file="vocab.json"):
    word_counter = Counter()
    def process_text(text):
        words = re.findall(r'\b[a-z]+\b', text.lower())
        word_counter.update(words)

    for split in ['train', 'validation']:
        for item in ds[split]:
            process_text(item['text'])

    most_common_words = dict(word_counter.most_common(vocab_size_limit))
    with open(output_file, "w") as f:
        json.dump(most_common_words, f, indent=4)
    
    print(f"Vocabulary saved to {output_file}")


create_reduced_vocab_and_save(ds, vocab_size_limit=5000, output_file="vocab.json")

# Tokenization class
class Vocab:
    def __init__(self, vocab_file):
        with open(vocab_file, "r") as f:
            vocab = json.load(f)

        words = list(vocab.keys())  # List of words
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2word = {idx: word for idx, word in enumerate(words)}
        self.vocab_size = len(words)

    def encode(self, text):
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return [self.word2idx[word] for word in words if word in self.word2idx]

    def decode(self, tokens):
        return " ".join([self.idx2word[token] for token in tokens if token in self.idx2word])

# instance of vocab class
vocab = Vocab("vocab.json")

# Tokenize the dataset
def tokenize_dataset(ds, vocab):
    def tokenize_item(item):
        tokenized_text = vocab.encode(item['text'])
        item['tokens'] = tokenized_text  # Store under 'tokens'
        return item

    # Apply tokenization to all items in the dataset
    tokenized_ds = ds.map(tokenize_item, batched=False, load_from_cache_file=False)
    return tokenized_ds

tokenized_ds = tokenize_dataset(ds, vocab)

# Concatenate all tokens into one long list
def concatenate_tokens(tokenized_ds):
    all_tokens = []
    for split in ['train', 'validation']:
        for item in tokenized_ds[split]:
            all_tokens.extend(item['tokens'])
    return all_tokens

all_tokens = concatenate_tokens(tokenized_ds)

# Define the TokenizedDataset class
class TokenizedDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens  # Concatenated list of tokens
        self.block_size = block_size

    def __len__(self):
        # Total number of sequences
        return (len(self.tokens) - self.block_size) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1  # +1 for target
        x = self.tokens[start_idx:start_idx + self.block_size]  # Context
        y = self.tokens[start_idx + 1:end_idx]  # Target

        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor

# Create the DataLoader
def create_dataloader(tokens, block_size=128, batch_size=4):
    dataset = TokenizedDataset(tokens, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

# Instantiate the DataLoader
block_size = 64 # Length of sequences
batch_size = 32
dataloader = create_dataloader(all_tokens, block_size=block_size, batch_size=batch_size)


for batch_idx, (x, y) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}: x shape {x.shape}, y shape {y.shape}")
    if batch_idx == 0:
        print(f"x[0]: {x[0]}")
        print(f"y[0]: {y[0]}")
    break 
