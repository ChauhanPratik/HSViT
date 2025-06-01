import torch
import torch.nn as nn
import torch.nn.functional as F

class TumorContextModule(nn.Module):
    """
    Learns anatomical priors via context bias and weak attention.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.conv_context = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, N, D] – Token embeddings
        """
        x_trans = x.transpose(1, 2)  # [B, D, N]
        context = self.conv_context(x_trans).transpose(1, 2)  # [B, N, D]
        gate = self.context_gate(x)  # [B, N, D]
        return x + context * gate