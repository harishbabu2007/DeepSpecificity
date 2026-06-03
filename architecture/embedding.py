import torch.nn as nn
import torch
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class ProteinEncoderWithPE(nn.Module):
    def __init__(self, in_dim, d_model):
        super(ProteinEncoderWithPE, self).__init__()

        self.token_embedding = nn.Linear(in_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)

    def forward(self, protein_features):
        return self.pos_encoder(self.token_embedding(protein_features))


class DNAEncoderWithPE(nn.Module):
    def __init__(self, in_dim, d_model):
        super(DNAEncoderWithPE, self).__init__()

        self.token_embedding = nn.Linear(in_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)

    def forward(self, dna_features):
        return self.pos_encoder(self.token_embedding(dna_features))