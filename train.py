import logging
import argparse
import random

import torch
import tqdm

from models import (BasicTransformer,
                    EnhancedAttentionTransformer,
                    DeepFeedForwardTransformer,
                    RMSNormTransformer)

from dataset import create_dataloader_from_file

DEFAULT_SEED = 42


def __init_randomness__(seed=DEFAULT_SEED):
    torch.manual_seed(seed)
    random.seed(seed)


def train(transformer_model=BasicTransformer, num_epochs=10, model_save_path='artifacts/model.pt'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device}")

    logging.info("Initialising the training process...")

    # TODO: Get vocab size and dataset. Assuming rn
    vocab_threshold = 5000
    train, validation = create_dataloader_from_file('roneneldan/TinyStories', 512, 0.0005, vocab_threshold, 16, 8)

    model = transformer_model(vocab_threshold)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    logging.debug("Starting to train the model")

    training_losses, validation_losses = [], []
    training_batch_losses, validation_batch_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        training_batch_loss = 0
        for batch in train:
            src, tgt = batch

            src = src.to(device)
            tgt = tgt.to(device)
            logging.debug(f"Moved src and tgt to {device}")

            optim.zero_grad()
            # NOTE: Padding token is 2 or now, can change later
            output = model(src, src == 2)
            loss = loss_fn(output, tgt, ignore_indexes=2)
            loss.backward()
            optim.step()

            training_batch_loss += loss.item()
            training_batch_losses.append(loss.item())
            # NOTE: find a way to show per batch loss with tqdm

        model.eval()
        validation_batch_loss = 0
        for batch in validation:
            src, tgt = batch

            src = src.to(device)
            tgt = tgt.to(device)

            # TODO: ensure that attn mask is correct!
            output = model(src, src == 0)
            loss = loss_fn(output, tgt)

            validation_batch_loss += loss.item()
            validation_batch_losses.append(loss.item())
            # NOTE: This could blow up very quickly, make sure that this
            # is fixed soon so that we dont have 4295498GB of artifacts
            # Ideally: save like 5 or smth in total
            torch.save(model.state_dict(), model_save_path + f".tmp.{epoch}")
        validation_losses.append(validation_batch_loss)
        training_losses.append(training_batch_loss)

    torch.save(model.state_dict(), model_save_path)

    print(f"Training Losses: {training_losses}")
    print(f"Validation Losses: {validation_losses}")

    print(f"Training Batch Losses: {training_batch_losses}")
    print(f"Validation Batch Losses: {validation_batch_losses}")


if __name__ == '__main__':
    avail_models = {
        'basic': BasicTransformer,
        'enhanced-attention': EnhancedAttentionTransformer,
        'deep-ff': DeepFeedForwardTransformer,
        'rmsnorm': RMSNormTransformer
    }

    # Create the parser
    parser = argparse.ArgumentParser(description="Train a transformer")

    # Allow for logging support throughout the app
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )

    parser.add_argument(
        '-s', '--seed',
        help="Force seed for deterministic randomness",
        action="store", dest="seed", default=DEFAULT_SEED, type=int
    )

    parser.add_argument(
        '-m', '--model',
        help="Choose what model to run", choices=avail_models.keys(),
        action='store', dest='model', default='basic'
    )

    parser.add_argument(
        '-p', '--model-save-path',
        help="Choose where is the model saved",
        action='store', dest='model_path', default='artifacts/model.pt'
    )

    # Parse the arguments
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    __init_randomness__(args.seed)

    model = avail_models[args.model]

    train(model, model_save_path = args.model_path)
