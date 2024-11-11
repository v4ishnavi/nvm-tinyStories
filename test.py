import argparse
import logging
import os
import json

import tokenizers
import torch
import torch.nn.functional as F

import config
from models import BasicTransformer
from dataset import create_dataloader_from_file


# TODO: Add in GenerationConfig
NUM_SENTENCES = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sentences using a trained transformer model.")

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
    parser.add_argument('config', help="Path to config [TOML] file that the model was trained with.")
    return parser.parse_args()


def load_model_and_vocab(configuration, device='cpu'):
    logging.info(f"Loading model from {configuration.transformer.path} and vocab from {configuration.tokenizer.path}...")
    if not os.path.exists(configuration.transformer.path + "/model.pt") or not os.path.exists(configuration.tokenizer.path):
        raise FileNotFoundError("Model/Tokenizer not found!")

    model = BasicTransformer(configuration)
    model.load_state_dict(
        torch.load(
            configuration.transformer.path + "/model.pt",
            map_location=device,
            weights_only=True
        ))
    model.to(device)
    model.eval()

    tokenizer = tokenizers.Tokenizer.from_file(configuration.tokenizer.path)

    return model, tokenizer


def generate_sentence(model, tokenizer, input_tensor, device='cpu', max_len=50):
    model.eval()
    input_tensor = input_tensor.to(device).unsqueeze(0)
    # TODO: Make sure padding token isn't hardcoded
    for _ in range(max_len):
        eos_token = torch.sum(input_tensor > 0)
        with torch.no_grad():
            output = model(input_tensor, (input_tensor == 0).float())
            output_token_logits = output[:, eos_token - 1, :]
            next_token_probs = F.softmax(output_token_logits, dim=-1)
            next_token = torch.argmax(next_token_probs, dim=-1).item()

            input_tensor[0, eos_token] = next_token

        # NOTE: Make sure that this is picked from the tokens. This is currently hardcoded
        if next_token == 2:
            break

    return input_tensor.squeeze(0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    logging.basicConfig(level=args.loglevel)
    configuration = config.read_config(args.config)
    if configuration.seed:
        torch.manual_seed(configuration.seed)

    configuration.transformer.load = True
    configuration.tokenizer.load = True
    model, tokenizer = load_model_and_vocab(configuration, device=device)
    _, val_loader = create_dataloader_from_file(configuration)

    completed_sentences = []
    idx = 0
    for src, tgt in val_loader:
        for sentence in src:
            idx += 1
            if idx > NUM_SENTENCES:
                break

            eos_pos = torch.sum(sentence > 0) # Count the number of non-padding tokens
            # NOTE: need to replace padding token placeholder with actual value!!
            prompt_tokens = sentence.clone()
            prompt_tokens[eos_pos // 2:] = 0  # NOTE: Ensure that padding is correct
            prompt = tokenizer.decode(prompt_tokens.tolist())

            # Generate the rest of the sentence
            # NOTE: Take max length from config!!!
            completed_sentence_tokens = generate_sentence(model, tokenizer, prompt_tokens, device=device, max_len=50)
            generated_sentence = tokenizer.decode(completed_sentence_tokens.tolist())

            # Format the result as prompt *** generated_sentence (paper follows this syntax)
            completed_sentences.append({
                "story": prompt,
                "completion": generated_sentence,
                "actual": tokenizer.decode(sentence[eos_pos // 2:eos_pos].tolist())
            })

        idx += 1
        if idx > NUM_SENTENCES:
            break

    print(json.dumps(completed_sentences))


if __name__ == "__main__":
    main()
