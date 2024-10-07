import torch
import json
import argparse
from models import BasicTransformer
from dataset import create_dataloader_from_file
import torch.nn.functional as F
import pickle
import logging

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
    parser.add_argument(
        '-m', '--model-path', 
        help="Path to the saved model file", 
        default="artifacts/model.pt"
    )
    parser.add_argument(
        '-vocab', '--vocab-path', 
        help="Path to the vocab file", 
        default="artifacts/vocab.pkl"
    )
    parser.add_argument(
        '--max-len', 
        help="Maximum length of the generated sentences", 
        type=int, 
        default=50
    )
    parser.add_argument(
        '--val-limit', 
        help="Limit the number of sentences to generate", 
        type=int, 
        default=16
    )
    parser.add_argument(
        '--output-file', 
        help="File where completed sentences will be saved", 
        default="completed_sentences.json"
    )

    parser.add_argument(
        "-s",
        "--seed",
        help="Force seed for deterministic randomness",
        action="store",
        dest="seed",
        type=int,
        default=42
    )
    return parser.parse_args()

def load_model_and_vocab(model_path, vocab_path):
    logging.info(f"Loading model from {model_path} and vocab from {vocab_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = BasicTransformer(vocab_size=5000) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    idx2word = {idx: word for word, idx in vocab.items()}
    word2idx = {word: idx for idx, word in idx2word.items()}
    
    logging.info("Model and vocab loaded successfully.")
    return model, device, idx2word, word2idx

def decode_sentence(tokens, idx2word):
    logging.debug(f"Decoding tokens: {tokens}")
    sentence = ' '.join([idx2word[token] for token in tokens if token in idx2word])
    logging.debug(f"Decoded sentence: {sentence}")
    return sentence

def generate_sentence(model, input_seq, word2idx, device, max_len=50):
    logging.info(f"Generating sentence with max length {max_len}...")
    model.eval()
    input_tensor = torch.tensor([input_seq], device=device)

    for step in range(max_len):
        with torch.no_grad():
            logging.debug(f"Generating token at step {step + 1}/{max_len}...")
            output = model(input_tensor, input_tensor) 
            output_token_logits = output[:, -1, :] 
            next_token_probs = F.softmax(output_token_logits, dim=-1)
            next_token = torch.argmax(next_token_probs, dim=-1).item()

        input_seq.append(next_token)
        input_tensor = torch.tensor([input_seq], device=device)

        if next_token == word2idx.get("<eos>"):
            logging.info(f"Generated <eos> token at step {step + 1}, stopping generation.")
            break

    return input_seq

def main():
    args = parse_args()
    
    logging.basicConfig(level=args.loglevel)
    
    if args.seed:
        logging.info(f"Setting random seed to {args.seed}")
        torch.manual_seed(args.seed)

    model, device, idx2word, word2idx = load_model_and_vocab(args.model_path, args.vocab_path)
    
    logging.info(f"Loading validation dataset with limit of {args.val_limit} sentences...")
    _, val_loader = create_dataloader_from_file("roneneldan/TinyStories", 512, 0.0005, 5000, 16, 8)

    completed_sentences = []
    for i, (src, tgt) in enumerate(val_loader):
        if i >= args.val_limit:
            logging.info(f"Reached the limit of {args.val_limit} sentences, stopping.")
            break

        for sentence in src:
            input_sentence = sentence.tolist()
            logging.debug(f"Input sentence tokens: {input_sentence[:10]}")
            
            completed_sentence_tokens = generate_sentence(model, input_sentence[:10], word2idx, device, max_len=args.max_len)
            completed_sentence = decode_sentence(completed_sentence_tokens, idx2word)
            
            logging.debug(f"Completed sentence: {completed_sentence}")
            completed_sentences.append(completed_sentence)

    logging.info(f"Saving completed sentences to {args.output_file}...")
    with open(args.output_file, "w") as outfile:
        json.dump(completed_sentences, outfile, indent=4)

    logging.info(f"Completed sentences have been saved to '{args.output_file}'.")

if __name__ == "__main__":
    main()
