
configs = [
    {'vocab_size': 1500, 'd_model': 512, 'num_encoder_layers': 8, 'nhead': 8, 'dim_feedforward': 4*512},
    {'vocab_size': 1500, 'd_model': 1024, 'num_encoder_layers': 8, 'nhead': 8, 'dim_feedforward': 4*1024},
    {'vocab_size': 1500, 'd_model': 768, 'num_encoder_layers': 8, 'nhead': 8, 'dim_feedforward': 4*768}
]

def calculate_encoder_parameters(vocab_size, d_model, num_encoder_layers, nhead, dim_feedforward):
    embedding_params = vocab_size * d_model
    mha_params = 4 * (d_model ** 2) + 4 * d_model
    ffnn_params = 2 * d_model * dim_feedforward + d_model + dim_feedforward
    layernorm_params = 2 * 2 * d_model
    final_ffn_params = d_model * vocab_size + vocab_size
    total_params = embedding_params + num_encoder_layers * (mha_params + ffnn_params + layernorm_params) + layernorm_params + final_ffn_params
    
    return total_params

# Loop through the configurations and calculate total parameters for each encoder-only configuration
for config in configs:
    total_params = calculate_encoder_parameters(config['vocab_size'], config['d_model'], config['num_encoder_layers'], config['nhead'], config['dim_feedforward'])
    print(f"Total number of parameters for encoder-only model with d_model {config['d_model']}: {total_params}")

