import torch
import torch.nn as nn

class DecoderHead(nn.Module):
    def __init__(self, embed_dim, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)     # class prediction
        self.bbox_regressor = nn.Linear(embed_dim, 4)           # bounding box: x, y, w, h
        self.boundary_decoder = nn.Sequential(                  # edge map head
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, D] CLS token or global pooled feature
        """
        class_logits = self.classifier(x)         # [B, C]
        bbox_coords = self.bbox_regressor(x)     # [B, 4]
        boundary_map = self.boundary_decoder(x)  # [B, 224*224]
        boundary_map = boundary_map.view(-1, 1, 224, 224)
        return class_logits, bbox_coords, boundary_map
