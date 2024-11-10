import logging
import argparse
import random
import os
import datetime

import torch
from tqdm import tqdm

from models import BasicTransformer
from dataset import create_dataloader_from_file
from config import read_config, save_model_with_config

DEFAULT_SEED = 42


def __init_randomness__(seed=DEFAULT_SEED):
    torch.manual_seed(seed)
    random.seed(seed)


def train(config, transformer_model=BasicTransformer, disable_progress_bars=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device}")

    logging.info("Initialising the training process...")

    train, validation = create_dataloader_from_file(config)
    model = transformer_model(config)

    if config.transformer.load:
        model.load_state_dict(torch.load(
            config.transformer.path + "/model.pt",
            map_location=device
        ))
    model = model.to(device)

    amp_availability = torch.amp.autocast_mode.is_autocast_available("cuda:0" if torch.cuda.is_available() else "cpu")
    if amp_availability:
        logging.info("Automatic Mixed Precision (AMP) scaling available! "
                     "Using that to improve training speeds")
        scaler = torch.amp.GradScaler(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    logging.info("Starting to train the model")

    training_losses, validation_losses = [], []
    training_batch_losses, validation_batch_losses = [], []
    for epoch in range(config.trainer.num_epochs):
        # Trainer Loop
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
            train_pbar.set_description(f"Epoch: {epoch + 1} | Batch loss: {loss.item()}")

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
            valid_pbar.set_description(f"Epoch: {epoch + 1} | Valid Batch loss: {loss.item()}")

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

    slurm_job_id = os.getenv("SLURM_JOB_ID")
    config.output.stdout_path = config.output.stdout_path.format(slurm_job_id)
    config.output.stderr_path = config.output.stderr_path.format(slurm_job_id)
    config.output.save_time = datetime.datetime.now()
    config.epochs_trained_for = config.trainer.num_epochs

    save_model_with_config(config, model)


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
        '-dp', '--disable-progress-bars',
        help='Disable progress bars. Useful during interactive jobs',
        action='store_const', dest="disable_progress_bars", const=True,
        default=False
    )

    parser.add_argument(
        'config',
        help="Provide the path to the config file [TOML]. For more " \
        "information, check config.py",
        action="store"
    )

    # Parse the arguments
    args = parser.parse_args()
    configuration = read_config(args.config)

    logging.basicConfig(level=args.loglevel)

    # We assume the mdoel to be basic
    # TODO: Add it to the TOML file
    model = avail_models['basic']

    __init_randomness__(configuration.seed)

    train(configuration, model, disable_progress_bars=args.disable_progress_bars)
