import math

import torch
import torch.nn as nn


class BiasedMultiheadAttention(nn.Module):
    """
    Multi-head cross attention with an additive attention bias.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, sigma=6.0, max_clamp=50.0):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "sigma",
            torch.tensor(float(sigma)),
        )
        self.max_clamp = max_clamp

    def forward(
        self,
        query,
        key,
        value,
        distance_matrix=None,
    ):

        B, Nq, _ = query.shape
        Nk = key.shape[1]

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)


        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # Q : (B,H,Nq,Dh)
        # K : (B,H,Nk,Dh)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1),
        )

        scores /= math.sqrt(self.head_dim)

        # Add Gaussian distance bias
        if distance_matrix is not None:
            distance_matrix = torch.clamp(
                distance_matrix,
                min=0.0,
                max=self.max_clamp,
            )

            gaussian_bias = -(
                distance_matrix ** 2
            ) / (2 * self.sigma * self.sigma)

            gaussian_bias = gaussian_bias.unsqueeze(1)

            scores = scores + gaussian_bias

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(
            attention,
            V,
        )

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, Nq, self.embed_dim)

        context = self.out_proj(context)

        return context
