from datetime import datetime
import dataclasses
import pathlib
import os
import argparse

import tosholi


@dataclasses.dataclass
class TransformerConfig:
    """
    This stores the configuration for the model itself. This includes
    number of layers, layer sizes, etc.
    """
    # What is the model dimension? Used throughout the transformer
    model_dimension: int = 512
    # How many heads should each transformer layer have?
    heads: int = 8
    # How many layers should the transformer have?
    layers: int = 6
    # What should be the dimension of the feedforward layer at the
    # end of every layer?
    feedforward_dimension: int = 2048

    # Load model from checkpoint, or
    load: bool = False
    # Where is the transformer stored?
    # This is the path to the folder, not the model itself. The model is
    # stored in model.pt inside this. The idea is to setup symlinks to all
    # needed paths (logs, tokenizers, etc) from inside this folder instead
    # of having 10 different folders that contain stuff
    path: str = 'artifacts/transformer'


@dataclasses.dataclass
class TokenizerConfig:
    """
    This contains the configuration for the vocabulary. It is assumed that
    this is correct and will not be questioned :) Any errors arised due to
    the disparity between this and the actual tokenizer are assumed to be
    out of scope.
    """
    # Load from file, or create new tokenizer?
    load: bool = False
    # Where to store/load tokenizer from?
    # This is the path to the tokenizer folder. The tokenizer is stored
    # as is by HuggingFace and we do not interfere in that regards.
    path: str = 'artifacts/tokenizer.json'
    # What is the vocabulary size of the tokenizer?
    # It is preferred that it is a multiple of 8 for faster training on
    # GPUs, since it allows utilization of the Tensor core
    vocab_size: int = 8192


@dataclasses.dataclass
class DataloaderConfig:
    """
    This contains the configuration for loading data.
    """
    # What HuggingFace dataset do we want to use?
    dataset: str = 'roneneldan/TinyStories'
    # How much data do we want to use?
    fraction_of_data: float = 0.01
    # What is the longest sentence we want to take? (measured in tokens)
    max_sentence_length: int = 512
    # Batch size used for training (adjust based on hardware)
    train_batch_size: int = 32
    # Batch size used for validation (adjust based on hardware)
    validation_batch_size: int = 16


@dataclasses.dataclass
class TrainerConfig:
    """
    This stores the configuration for training (think: number of epochs,
    learning rate etc.)
    """
    learning_rate: float = 1e-3
    num_epochs: int = 4


@dataclasses.dataclass
class NeptuneConfig:
    """
    This stores the configuration for Neptune Logging. Note that the API
    key is obtained from NEPTUNE_API_TOKEN
    """
    project_name: str = 'mon/ANLP'
    run_name: str = 'training-run'
    # TODO: Debug this to ensure this works as expected
    # tags: tuple = ('train', 'debug-run')


@dataclasses.dataclass
class OutputConfig:
    """
    This option is not read, only output, to reference the logs and other
    possible artifacts that are generated during a training run. This
    will be ignored (overwritten, really) while reading a file in. This
    is primarily for storing output data while saving the model.
    """
    # Where are the training logs stored?
    stdout_path: str = '.sbatch-training-logs.out.{}'
    stderr_path: str = '.sbatch-training-logs.err.{}'
    epochs_trained_for: int = 0
    save_time: datetime = datetime.now()


@dataclasses.dataclass
class Config:
    transformer: TransformerConfig
    tokenizer: TokenizerConfig
    dataloader: DataloaderConfig
    trainer: TrainerConfig
    output: OutputConfig
    neptune: NeptuneConfig

    # What random seed to use?
    seed: int = 42


def read_config(path):
    with open(path, 'rb') as f:
        config = tosholi.load(Config, f)

    return config


def save_model_with_config(config, model):
    import torch
    # This will save a model along with the correspoding config file
    # This ensures that one is able to load a model as-is without issues

    # TODO: Check if model exists, and do not overwrite? ig
    save_path = pathlib.Path(config.transformer.path)

    # Create directory along with parents, also ignore if already
    # exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model into model.pt (assumes model is a PyTorch-like model)
    torch.save(model.state_dict(), str(save_path / "model.pt"))

    # symlink the tokenizer
    tokenizer_path = save_path / "tokenizer.json"
    rel_tokenizer_path = os.path.relpath(config.tokenizer.path, start=save_path)
    tokenizer_path.symlink_to(rel_tokenizer_path)

    # symlinks to output logs to allow for easy discoverability
    stdout_path = (save_path / "stdout_log")
    rel_stdout_path = os.path.relpath(config.output.stdout_path, start=save_path)
    stdout_path.symlink_to(rel_stdout_path)

    stderr_path = (save_path / "stderr_log")
    rel_stderr_path = os.path.relpath(config.output.stderr_path, start=save_path)
    stderr_path.symlink_to(rel_stderr_path)

    # save the updated config for sentence and report generation
    with (save_path / "config.toml").open(mode='wb') as f:
        tosholi.dump(config, f)


def save_config(path):
    """
    Saves the configuration to the given path. Can be useful for CLI
    configuration generation, not currently used.
    """
    # TODO: How can we configure it as a config generator?
    config = Config(
        transformer=TransformerConfig(),
        tokenizer=TokenizerConfig(),
        dataloader=DataloaderConfig(),
        trainer=TrainerConfig(),
        output=OutputConfig(),
        neptune=NeptuneConfig(),
    )

    with open(path, 'wb') as f:
        tosholi.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save default config")
    parser.add_argument(
        'path',
        help="Path to store config file"
    )
    args = parser.parse_args()
    save_config(args.path)
