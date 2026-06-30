import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, dna_channels, hidden_size=64):
        super().__init__()

        self.conv1 = nn.Conv1d(
            dna_channels, hidden_size, kernel_size=3, stride=1, padding=1
        )

        self.conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # x: (B, N, C) not B,C,N

        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.permute(0, 2, 1)

        return x
