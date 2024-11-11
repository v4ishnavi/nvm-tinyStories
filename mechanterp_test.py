import logging
import argparse
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from models import BasicTransformer
from config import read_config
import json


class AttentionVisualizer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # the toml file stores the tokenizer, so this opens the path 
        tokenizer_path = config.tokenizer.path
        logging.info(f"Loading tokenizer from {tokenizer_path}...")
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)

        self.vocab = {
            token['content']: token['id']
            for token in tokenizer_data.get('added_tokens', [])
        }
        # update the vocab with the model vocab,, this is a dictionary
        self.vocab.update(tokenizer_data.get('model', {}).get('vocab', {}))

        self.pad_token = tokenizer_data['padding']['pad_token']
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        self.unk_token = tokenizer_data['model']['unk_token']
        self.max_length = tokenizer_data['truncation']['max_length']

        self.idx2token = {idx: token for token, idx in self.vocab.items()}

        # Load model
        logging.info("Loading model checkpoint...")
        model_path = config.transformer.path + "/model.pt"
        self.model = BasicTransformer(config)
        # print state dict to check if the model exists...
        # print(self.model.state_dict()) 
        # seems to exist..
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

        # how to check what is in the model? - print the model 
        print(self.model)
        self.model.to(self.device)
        self.model.eval()
        
        # Store attention weights
        self.attention_weights = None

    def preprocess_text(self, text):
        """Preprocess input text based on tokenizer."""
        tokens = text.lower().split()
        tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Convert tokens to indices
        indices = [self.vocab.get(token, self.vocab.get(self.unk_token)) for token in tokens]
        
        # Pad sequence
        if len(indices) < self.max_length:
            indices += [self.vocab[self.pad_token]] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
            
        return torch.tensor(indices).unsqueeze(0).to(self.device), tokens

    def get_attention_matrix(self, layer_idx, head_idx):
        """Extract attention matrix for specified layer and head."""
        def hook(module, input, output):
            # MultiheadAttention outputs attention weights as part of output tuple
            if isinstance(output, tuple) and len(output) > 1:
                logging.debug(f"Output printing: {output}")
                self.attention_weights = output[0].detach()  # Standard PyTorch MultiheadAttention behavior
                logging.debug(f"Attention weights shape added...")
                logging.debug(f"Attention weights shape: {self.attention_weights.shape}")
            else:
                raise ValueError("Attention weights not found in module output. Check model implementation.")
        return hook

    def visualize_attention(self, text, layer, head, image_path = None):
        logging.info(f"Visualizing attention for text: '{text}'")

        # Preprocess the input text
        input_ids, tokens = self.preprocess_text(text)
        padding_mask = (input_ids == self.vocab[self.pad_token]).to(dtype=torch.bool, device=self.device)

        logging.debug(f"Input shape: {input_ids.shape}, Padding mask shape: {padding_mask.shape}")

        # Register hook to capture attention weights
        layer_module = self.model.transformer_encoder.layers[layer].self_attn
        hook = layer_module.register_forward_hook(self.get_attention_matrix(layer, head))

        # Forward pass through the model
        try:
            with torch.no_grad():
                logging.debug("Starting forward pass...")
                self.model(input_ids, padding_mask)
                logging.debug("Forward pass completed.")
        finally:
            # Ensure hook is removed even if forward pass fails
            hook.remove()

        if self.attention_weights is None:
            logging.error("No attention weights captured. Verify model architecture and hook attachment.")
            raise ValueError("No attention weights captured. Check model architecture.")

        # Handle three-dimensional attention weights (batch, source, target)
        attn_weights = self.attention_weights[0, :len(tokens), :len(tokens)].cpu()  # Removed head dimension
        logging.debug(f"Attention weights shape: {attn_weights.shape}")

        # Generate heatmap for visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights.numpy(),
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap="viridis")
        plt.title(f"Attention Layer {layer}, Head {head}")
        plt.xlabel("Target Tokens")
        plt.ylabel("Source Tokens")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        if image_path is not None:
            plt.savefig(image_path)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize transformer attention patterns")
    parser.add_argument('-d', '--debug', help="Enable debugging logs", 
                       action='store_const', const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('--config', required=True, help="Path to config TOML file")
    parser.add_argument('--text', required=True, help="Text input for visualization")
    parser.add_argument('--layer', type=int, default=0, help="Layer index to visualize")
    parser.add_argument('--head', type=int, default=0, help="Head index to visualize")
    parser.add_argument('--all-heads', type = bool, default = False, help="Print attention matrices for all heads in the specified layer")
    parser.add_argument('-v', '--verbose', help="Enable verbose logs", 
                       action='store_const', const=logging.INFO, default=logging.WARNING)
    # parser.add_argument('--all-heads', action='store_true', help="Print attention matrices for all heads in the specified layer")
    parser.add_argument('--image-path',help="Path to save the image")
    args = parser.parse_args()

    logging.basicConfig(level=min(args.debug, args.verbose))

    config = read_config(args.config)
    visualizer = AttentionVisualizer(config)
    if args.all_heads:
        for head in range(config.transformer.heads):
            visualizer.visualize_attention(args.text, args.layer, head, args.image_path)
    else:
        visualizer.visualize_attention(args.text, args.layer, args.head, args.image_path)
    
if __name__ == '__main__':
    main()