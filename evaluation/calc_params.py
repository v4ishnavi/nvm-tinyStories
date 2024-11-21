
configs = [
    {'name': 'ANLP-72', 'vocab_size': 8192, 'd_model': 512, 'num_encoder_layers': 6, 'nhead': 8, 'dim_feedforward': 2048},
    {'name': 'ANLP-27', 'vocab_size': 8192, 'd_model': 512, 'num_encoder_layers': 6, 'nhead': 8, 'dim_feedforward': 2048},
    {'name': 'ANLP-24', 'vocab_size': 8192, 'd_model': 1024, 'num_encoder_layers': 6, 'nhead': 8, 'dim_feedforward': 2048},
    {'name': 'ANLP-23', 'vocab_size': 8192, 'd_model': 256, 'num_encoder_layers': 6, 'nhead': 8, 'dim_feedforward': 2048},
    {'name': 'ANLP-65', 'vocab_size': 8192, 'd_model': 768, 'num_encoder_layers': 12, 'nhead': 8, 'dim_feedforward': 2048},

]

def calculate_encoder_parameters(config, vocab_size, d_model, num_encoder_layers, nhead, dim_feedforward):
    print(f"The run is: {config['name']} which has the following configuration: {config}")
    embedding_params = vocab_size * d_model
    mha_params = 4 * (d_model ** 2) + 4 * d_model
    ffnn_params = 2 * d_model * dim_feedforward + d_model + dim_feedforward
    layernorm_params = 2 * 2 * d_model
    final_ffn_params = d_model * vocab_size + vocab_size
    total_params = embedding_params + num_encoder_layers * (mha_params + ffnn_params + layernorm_params) + layernorm_params + final_ffn_params
    
    return total_params

# Loop through the configurations and calculate total parameters for each encoder-only configuration
for config in configs:
    total_params = calculate_encoder_parameters(config, config['vocab_size'], config['d_model'], config['num_encoder_layers'], config['nhead'], config['dim_feedforward'])
    print(f"Total number of parameters for encoder-only model with d_model {config['d_model']}: {total_params}")

