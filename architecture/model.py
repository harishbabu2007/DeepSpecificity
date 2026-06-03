import torch.nn as nn
import torch
from architecture.embedding import ProteinEncoderWithPE, DNAEncoderWithPE 


class MainModel(nn.Module):
    def __init__(self,
                 len_dna_features: int,
                 len_prot_features: int,
                 d_model: int,
                 n_head_dna: int,
                 n_enc_dna: int,
                 n_head_prot: int,
                 n_enc_prot: int,
                 n_cross_att_heads: int,
                 n_enc_pwm: int,
                 n_head_pwm: int):

        super(MainModel, self).__init__()

        self.protein_embedder = ProteinEncoderWithPE(len_prot_features, d_model)
        self.dna_embedder = DNAEncoderWithPE(len_dna_features, d_model)

        self.encoder_layer_dna = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head_dna, batch_first=True)
        self.transformer_encoder_dna = nn.TransformerEncoder(self.encoder_layer_dna, num_layers=n_enc_dna)

        self.encoder_layer_protein = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head_prot, batch_first=True)
        self.transformer_encoder_protein = nn.TransformerEncoder(self.encoder_layer_protein, num_layers=n_enc_prot)

        self.protein_to_dna_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_cross_att_heads,
            batch_first=True
            )

        self.dna_to_protein_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_cross_att_heads,
            batch_first=True
        )

        self.protein_cross_norm = nn.LayerNorm(d_model)
        self.dna_cross_norm = nn.LayerNorm(d_model)

        self.pwm_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head_pwm,
                batch_first=True
            ),
            num_layers=n_enc_pwm
        )

        self.pwm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 4),
        )

    def forward(self, dna_features, protein_features):
        protein_embedding = self.protein_embedder(protein_features)
        dna_embedding = self.dna_embedder(dna_features)

        protein_embedding = self.transformer_encoder_protein(protein_embedding)
        dna_embedding = self.transformer_encoder_dna(dna_embedding)

        protein_before_cross = protein_embedding
        dna_before_cross = dna_embedding

        protein_context, _ = self.protein_to_dna_attention(
            query=protein_before_cross,
            key=dna_before_cross,
            value=dna_before_cross
        )
        protein_embedding = protein_embedding + protein_context

        protein_embedding = self.protein_cross_norm(
            protein_embedding + protein_context
        )

        dna_context, _ = self.dna_to_protein_attention(
            query=dna_before_cross,
            key=protein_before_cross,
            value=protein_before_cross
        )
        dna_embedding = dna_embedding + dna_context

        dna_embedding = self.dna_cross_norm(
            dna_embedding + dna_context
        )

        dna_embedding = self.pwm_encoder(dna_embedding)

        pwm_logits = self.pwm_head(dna_embedding)

        return pwm_logits
