"""
CNN + LSTM model for excavator activity recognition.

Architecture:
  - CNN backbone (ResNet18 pretrained) → per-frame feature vector (512-d)
  - LSTM over a sequence of frames → temporal context
  - FC head → 4 class logits
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNEncoder(nn.Module):
    def __init__(self, feature_dim: int = 512, freeze_backbone: bool = False):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC layer; keep avgpool → 512-d output
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = feature_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return self.backbone(x).flatten(1)  # (B, 512)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.encoder = CNNEncoder(feature_dim=feature_dim, freeze_backbone=freeze_backbone)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Encode each frame independently
        features = self.encoder(x.view(B * T, C, H, W))   # (B*T, 512)
        features = features.view(B, T, -1)                  # (B, T, 512)
        # LSTM over time
        out, _ = self.lstm(features)                        # (B, T, hidden)
        # Use last timestep for classification
        logits = self.classifier(out[:, -1, :])             # (B, num_classes)
        return logits
