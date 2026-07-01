import torch.nn as nn
import torch
from architecture.embedding import ProteinEncoderWithPE, DNAEncoderWithPE
from architecture.cnn_block import CNNBlock
from architecture.biased_attention import BiasedMultiheadAttention

class DeepSpecificityWithShape(nn.Module):
    def __init__(
        self,
        len_dna_features: int,
        len_dna_shape_features: int,
        len_prot_features: int,
        d_model: int,
        n_head_dna: int,
        n_enc_dna: int,
        n_head_prot: int,
        n_enc_prot: int,
        n_cross_att_heads: int,
        n_enc_pwm: int,
        n_head_pwm: int,
    ):

        super(DeepSpecificityWithShape, self).__init__()
        self.dropout_rate = 0.2

        self.protein_embedder = ProteinEncoderWithPE(len_prot_features, d_model)
        self.dna_embedder = DNAEncoderWithPE(len_dna_features, d_model)
        self.dna_shape_embedder = DNAEncoderWithPE(len_dna_shape_features, d_model)

        self.dna_input_norm = nn.LayerNorm(len_dna_features)
        self.shape_input_norm = nn.LayerNorm(len_dna_shape_features)
        self.protein_input_norm = nn.LayerNorm(len_prot_features)

        # all encoder layers
        self.encoder_layer_dna = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head_dna, batch_first=True, norm_first=True
        )
        self.transformer_encoder_dna = nn.TransformerEncoder(
            self.encoder_layer_dna, num_layers=n_enc_dna
        )

        self.encoder_layer_protein = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head_prot, batch_first=True, norm_first=True
        )
        self.transformer_encoder_protein = nn.TransformerEncoder(
            self.encoder_layer_protein, num_layers=n_enc_prot
        )

        self.encoder_layer_dna_shape = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head_dna//2, batch_first=True, norm_first=True
        )
        self.transformer_encoder_dna_shape = nn.TransformerEncoder(
            self.encoder_layer_dna_shape, num_layers=n_enc_dna//2
        )

        # all cross attentions
        # self.dna_to_protein_attention = nn.MultiheadAttention(
        #     embed_dim=d_model, num_heads=n_cross_att_heads, batch_first=True
        # )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.scale = d_model ** -0.5

        # self.dna_to_dna_shape_attention = nn.MultiheadAttention(
        #     embed_dim=d_model, num_heads=n_cross_att_heads, batch_first=True
        # )
        self.dna_to_protein_attention = BiasedMultiheadAttention(
            embed_dim=d_model,
            num_heads=n_cross_att_heads,
            dropout=self.dropout_rate,
        )

        self.dna_cross_norm = nn.LayerNorm(d_model)
        # self.dna_dna_shape_cross_norm = nn.LayerNorm(d_model)

        self.dna_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(self.dropout_rate),
        )

        # self.dna_mlp_2 = nn.Sequential(
        #     nn.Linear(d_model, d_model * 2),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout_rate),
        #     nn.Linear(d_model * 2, d_model),
        #     nn.Dropout(self.dropout_rate),
        # )

        self.dna_mlp_norm = nn.LayerNorm(d_model)
        # self.dna_mlp_norm_2 = nn.LayerNorm(d_model)

        self.dna_shape_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        # final pwm encoder and mlp
        self.pwm_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head_pwm, batch_first=True, norm_first=True
            ),
            num_layers=n_enc_pwm,
        )

        self.hidden_size = d_model

        self.conv_block = CNNBlock(d_model, self.hidden_size)

        self.pwm_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size, 4),
        )

    def forward(
        self,
        dna_features,
        dna_shape_features,
        protein_features,
        distance_matrix,
    ):
        dna_features = self.dna_input_norm(dna_features)
        dna_shape_features = self.shape_input_norm(dna_shape_features)
        protein_features = self.protein_input_norm(protein_features)

        protein_embedding = self.protein_embedder(protein_features)
        dna_embedding = self.dna_embedder(dna_features)
        dna_shape_embedding = self.dna_shape_embedder(dna_shape_features)

        protein_embedding = self.transformer_encoder_protein(protein_embedding)
        dna_embedding = self.transformer_encoder_dna(dna_embedding)
        dna_shape_embedding = self.transformer_encoder_dna_shape(dna_shape_embedding)

        dna_context = self.dna_to_protein_attention(
            query=dna_embedding,
            key=protein_embedding,
            value=protein_embedding,
            distance_matrix=distance_matrix,
        )

        dna_embedding = self.dna_cross_norm(
            dna_embedding + dna_context
        )

        dna_embedding = self.dna_mlp_norm(
            dna_embedding + self.dna_mlp(dna_embedding)
        )

        dna_embedding = torch.cat(
            [
                dna_embedding,
                dna_shape_embedding,
            ],
            dim=-1,
        )

        dna_embedding = self.dna_shape_fusion(
            dna_embedding
        )

        dna_embedding = self.pwm_encoder(dna_embedding)
        dna_embedding = dna_embedding + self.conv_block(dna_embedding)
        pwm_logits = self.pwm_head(dna_embedding)

        return pwm_logits
