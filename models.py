import torch
import torch.nn as nn
import torch.nn.init as init
import math


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class BasicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super(BasicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, attention_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        src_key_padding_mask = (attention_mask == 0).to(src.device)

        output = self.transformer_decoder(
            src.transpose(0, 1),
            tgt_key_padding_mask=src_key_padding_mask, is_causal=True)
        output = self.linear(output.mean(dim=0))
        return output


class EnhancedAttentionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=16, num_layers=6,
                 dim_feedforward=2048):
        super(EnhancedAttentionTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead,
                                                    dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 2)

    def forward(self, src, attention_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        src_key_padding_mask = (attention_mask == 0).to(src.device)

        output = self.transformer_encoder(
            src.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask)
        output = self.linear(output.mean(dim=0))
        return output


class DeepFeedForwardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super(DeepFeedForwardTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(nn.TransformerEncoderLayer(d_model,
                                                             nhead,
                                                             dim_feedforward))
            encoder_layers.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(0.1)
            ))
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, src, attention_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        src_key_padding_mask = (attention_mask == 0).to(src.device)

        output = src.transpose(0, 1)
        for layer in self.transformer_encoder:
            if isinstance(layer, nn.TransformerEncoderLayer):
                output = layer(output, src_key_padding_mask=src_key_padding_mask)
            else:
                output = output + layer(output)  # residual connection
        output = self.norm(output)
        output = self.linear(output.mean(dim=0))
        return output


class RMSNormTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super(RMSNormTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(nn.MultiheadAttention(d_model, nhead))
            encoder_layers.append(RMSNorm(d_model))
            encoder_layers.append(nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model)
            ))
            encoder_layers.append(RMSNorm(d_model))
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 2)

    def forward(self, src, attention_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        src_key_padding_mask = (attention_mask == 0).to(src.device)

        output = src.transpose(0, 1)
        for i, layer in enumerate(self.transformer_encoder):
            if isinstance(layer, nn.MultiheadAttention):
                output = layer(output, output, output, key_padding_mask=src_key_padding_mask)[0]
            else:
                output = layer(output)
        output = self.linear(output.mean(dim=0))
        return output
