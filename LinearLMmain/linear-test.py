import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.linear_model import LinearModel

def parse_args():
    parser = argparse.ArgumentParser(description="Generate sentences using linear models.")
    parser.add_argument('--conf', default='configs/config_linear.json', 
                        help="Path to config JSON file")
    return parser.parse_args()

def load_model_and_tokenizer(config, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    
    context_length = config['context_length']
    activation = 'relu' if config.get('use_relu') else 'none'
    
    model = LinearModel(
        num_tokens=len(tokenizer)+1, 
        T=context_length, 
        d=config['d'], 
        activation=activation
    )

    if config.get('checkpoint_path'):
        checkpoint = torch.load(config['checkpoint_path'], map_location=device)
        print("Checkpoint keys:", checkpoint.keys())
        print("Checkpoint mask shape:", checkpoint.get('mask', torch.tensor([])).shape)
        print("Checkpoint in_embedding shape:", checkpoint.get('in_embedding.weight', torch.tensor([])).shape)
        
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model, tokenizer

def generate_text(config, model, tokenizer, src, max_len=None, temperature=None):
    device = next(model.parameters()).device
    
    # Use first half of the source sentence as prompt
    eos_pos = torch.sum(src > 0)
    half_length = eos_pos // 2
    prompt_tokens = src[:half_length]
    actual_tokens = src[half_length:eos_pos]
    
    input_tensor = prompt_tokens.to(device).unsqueeze(0)
    max_len = max_len or input_tensor.shape[1] + 20
    temperature = temperature or 0.5

    for _ in range(max_len - input_tensor.shape[1]):
        eos_pos = torch.sum(input_tensor > 0)
        
        with torch.no_grad():
            output = model(input_tensor)
            output_token_logits = output[0, eos_pos - 1, :]
            output_token_logits = output_token_logits / temperature
            next_token_probs = F.softmax(output_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
        
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    return {
        "story": tokenizer.decode(prompt_tokens.tolist()),
        "actual": tokenizer.decode(actual_tokens.tolist()),
        "completion": tokenizer.decode(input_tensor.squeeze(0).tolist())
    }

def main():
    args = parse_args()
    with open(args.conf) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(config, device)

    # Load dataset (assuming your dataset loader returns tensors)
    from dataset import create_dataloader_from_file
    _, val_loader = create_dataloader_from_file(config)

    completed_sentences = []
    for src, _ in val_loader:
        for sentence in src:
            result = generate_text(config, model, tokenizer, sentence)
            completed_sentences.append(result)
            
            if len(completed_sentences) >= 200:
                break
        
        if len(completed_sentences) >= 200:
            break

    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'generated_sentences_{os.path.basename(args.conf)}.json')
    
    with open(output_filename, 'w') as f:
        json.dump(completed_sentences, f, indent=2)
    
    print(f"Generated sentences saved to {output_filename}")

if __name__ == "__main__":
    main()