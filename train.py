import logging
import argparse
import random

import torch
from tqdm import tqdm

from models import BasicTransformer

from dataset import create_dataloader_from_file

DEFAULT_SEED = 42

def __init_randomness__(seed=DEFAULT_SEED):
    torch.manual_seed(seed)
    random.seed(seed)


def train(transformer_model=BasicTransformer, num_epochs=3, model_save_path='artifacts/model.pt',
          load_model_checkpoint=None, disable_progress_bars=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device}")

    logging.info("Initialising the training process...")

    # TODO: Get vocab size and dataset. Asssuming rn
    vocab_size = 8192
    train, validation = create_dataloader_from_file('roneneldan/TinyStories', 512, 0.005, 16, 16, vocab_size)

    model = transformer_model(vocab_size)

    if load_model_checkpoint is not None:
        model.load_state_dict(torch.load(load_model_checkpoint, map_location=device))

    model = model.to(device)

    amp_availability = torch.amp.autocast_mode.is_autocast_available("cuda:0" if torch.cuda.is_available() else "cpu")
    if amp_availability:
        logging.info("Automatic Mixed Precision (AMP) scaling available! Using that to improve training speeds")
        scaler = torch.amp.GradScaler(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
    # NOTE: Padding token is 2 or now, can change later
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=2)

    logging.info("Starting to train the model")

    training_losses, validation_losses = [], []
    training_batch_losses, validation_batch_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        training_batch_loss = 0

        train_pbar = tqdm(train, disable=disable_progress_bars)
        valid_pbar = tqdm(validation, disable=disable_progress_bars)

        for batch in train_pbar:
            src, tgt = batch

            src = src.to(device)
            tgt = tgt.to(device)
            logging.debug(f"Moved src and tgt to {device}")

            optim.zero_grad(set_to_none=True)

            if amp_availability:
                # NOTE: we only run mixed precision training
                # since the biggest slowdown arrives during the backward
                # pass. How do we speed this up?
                with torch.amp.autocast('cuda'):
                    output = model(src, (src == 0).float())
                    loss = loss_fn(output.transpose(-1, -2), tgt)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                output = model(src, (src == 0).float())
                loss = loss_fn(output.transpose(-1, -2), tgt)
                loss.backward()
                optim.step()

            training_batch_loss += loss.item()
            training_batch_losses.append(loss.item())
            # NOTE: find a way to show per batch loss with tqdm
            train_pbar.set_description(f"Epoch: {epoch} | Batch loss: {loss.item()}")

        model.eval()
        validation_batch_loss = 0
        for batch in valid_pbar:
            src, tgt = batch

            src = src.to(device)
            tgt = tgt.to(device)

            # NOTE: Padding token is 2 or now, can change later
            output = model(src, src == 0)
            loss = loss_fn(output.transpose(-1, -2), tgt)
            validation_batch_loss += loss.item()
            validation_batch_losses.append(loss.item())
            valid_pbar.set_description(f"Epoch: {epoch} | Valid Batch loss: {loss.item()}")
        # NOTE: This could blow up very quickly, make sure that this
        # is fixed soon so that we dont have 4295498GB of artifacts
        # Ideally: save like 5 or smth in total
        # torch.save(model.state_dict(), model_save_path + f".tmp.{epoch}")
        validation_losses.append(validation_batch_loss)
        training_losses.append(training_batch_loss)

        print(f"Training Losses: {training_losses}")
        print(f"Validation Losses: {validation_losses}")

        print(f"Training Batch Losses: {training_batch_losses}")
        print(f"Validation Batch Losses: {validation_batch_losses}")

    torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    avail_models = {
        'basic': BasicTransformer
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

    parser.add_argument(
        '-l', '--model-checkpoint-load',
        help='Load model from checkpoint',
        action='store', dest='checkpoint', default=None
    )

    parser.add_argument(
        '-dp', '--disable-progress-bars',
        help='Disable progress bars. Useful during interactive jobs',
        action='store_const', dest="disable_progress_bars", const=True,
        default=False
    )

    # Parse the arguments
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    __init_randomness__(args.seed)

    model = avail_models[args.model]

    train(model, model_save_path=args.model_path, load_model_checkpoint=args.checkpoint, disable_progress_bars=args.disable_progress_bars)
