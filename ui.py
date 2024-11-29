import gradio as gr

import argparse
import logging
import os
import json

import tokenizers
import torch
# import torch.nn.functional as F

import config
from models import BasicTransformer
from dataset import create_dataloader_from_file


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
    
    def generate_sentence(input_tensor, model=model, tokenizer=tokenizer, device=device, context_len=512):
        input_tensor = torch.tensor(tokenizer.encode(input_tensor).ids)
        max_len = context_len - torch.sum(input_tensor > 0)
        model.eval()
        input_tensor = input_tensor.to(device).unsqueeze(0)
        # TODO: Make sure padding token isn't hardcoded
        for _ in range(max_len):
            eos_token = torch.sum(input_tensor > 0)
            with torch.no_grad():
                output = model(input_tensor, (input_tensor == 0).float())
                output_token_logits = output[:, eos_token - 1, :]
                next_token_probs = torch.nn.functional.softmax(output_token_logits, dim=-1)
                next_token = torch.argmax(next_token_probs, dim=-1).item()

                input_tensor[0, eos_token] = next_token
                yield tokenizer.decode(input_tensor[0].tolist())
            # NOTE: Make sure that this is picked from the tokens. This is currently hardcoded
            if next_token == 2:
                break

        return input_tensor.squeeze(0)

    demo = gr.Interface(
        fn=generate_sentence,
        inputs=["text"],
        outputs=["text"],
    )

    demo.launch()

if __name__ == "__main__":
    main()