import torch
import torch.nn as nn
from einops import rearrange

from hsvit.modules.pre_encoder import PreEncoder
from hsvit.modules.clar import CrossLayerAttentionRefinement
from hsvit.modules.token_filter import ProgressiveTokenFilter
from hsvit.modules.tcm import TumorContextModule
from hsvit.modules.decoder_head import DecoderHead

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_dim=128, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = None  # initialized in forward

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)                             # [B, emb_dim, H/patch, W/patch]
        x = rearrange(x, 'b c h w -> b (h w) c')     # [B, N, emb_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)        # [B, N+1, emb_dim]

        # Initialize positional embeddings dynamically
        if self.pos_embed is None or self.pos_embed.shape[1] != x.size(1):
            self.pos_embed = nn.Parameter(torch.randn(1, x.size(1), self.emb_dim).to(x.device))

        x = x + self.pos_embed
        return x


class ViTBackbone(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_dim=128, depth=4, heads=4,
                 use_clar=True, use_tcm=True, use_boundary_head=True):
        super().__init__()

        self.use_clar = use_clar
        self.use_tcm = use_tcm
        self.use_boundary_head = use_boundary_head

        self.pre_encoder = PreEncoder()
        self.patch_embed = PatchEmbedding(1, patch_size, emb_dim, img_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads, batch_first=True)
        self.transformer_blocks = nn.ModuleList([encoder_layer for _ in range(depth)])

        if use_clar:
            self.clar = CrossLayerAttentionRefinement(emb_dim=emb_dim, num_layers=depth)
        if use_tcm:
            self.tcm = TumorContextModule(embed_dim=emb_dim)

        self.token_filter = ProgressiveTokenFilter(keep_ratio=0.7)
        self.decoder = DecoderHead(embed_dim=emb_dim, num_classes=3)


    def forward(self, x):
        x = self.pre_encoder(x)
        x = self.patch_embed(x)

        hidden_states = []
        for layer in self.transformer_blocks:
            x = layer(x)
            hidden_states.append(x)

        if self.use_clar:
            x = self.clar(hidden_states)
        if self.token_filter:
            x = self.token_filter(x)
        if self.use_tcm:
            x = self.tcm(x)

        cls_token = x[:, 0]
        if self.use_boundary_head:
            class_logits, bbox_coords, boundary_map = self.decoder(cls_token)
        else:
            class_logits, bbox_coords, _ = self.decoder(cls_token)
            boundary_map = torch.zeros_like(bbox_coords[:, :1])  # dummy

        return class_logits, bbox_coords, boundary_map
