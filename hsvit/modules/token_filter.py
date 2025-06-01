import torch
import torch.nn as nn

class ProgressiveTokenFilter(nn.Module):
    """
    Retains top-k tokens dynamically based on importance scores.
    """
    def __init__(self, keep_ratio=0.7):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x):
        """
        x: [B, N, D] – Token sequence
        """
        B, N, D = x.shape
        cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]  # Separate CLS
        scores = patch_tokens.norm(dim=-1)  # [B, N-1]
        k = max(1, int(self.keep_ratio * patch_tokens.shape[1]))
        topk_indices = torch.topk(scores, k, dim=1).indices  # [B, k]

        batch_indices = torch.arange(B).unsqueeze(1).to(x.device)
        filtered = patch_tokens[batch_indices, topk_indices]  # [B, k, D]

        return torch.cat([cls_token, filtered], dim=1)