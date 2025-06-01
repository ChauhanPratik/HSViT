import torch
import torch.nn as nn

class CrossLayerAttentionRefinement(nn.Module):
    """
    Aggregates attention features from shallow and deep transformer layers.
    """
    def __init__(self, emb_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.query_proj = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(num_layers)])
        self.key_proj = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(num_layers)])
        self.value_proj = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(num_layers)])
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, hidden_states):
        """
        hidden_states: List of [B, N, D] from each transformer layer
        """
        B, N, D = hidden_states[0].shape
        fused = torch.zeros(B, N, D, device=hidden_states[0].device)

        for i in range(self.num_layers):
            q = self.query_proj[i](hidden_states[i])
            k = self.key_proj[i](hidden_states[i])
            v = self.value_proj[i](hidden_states[i])

            attn_weights = self.softmax(torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5))
            attn_out = torch.bmm(attn_weights, v)
            fused += attn_out / self.num_layers

        return self.output_proj(fused)