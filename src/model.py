"""Modelo baseline multimodal: encoder de imagem (timm) + encoder de texto
(transformers), com fusão simples por concatenação antes da cabeça final."""

import timm
import torch
import torch.nn as nn
from transformers import AutoModel

from . import config


class ImageEncoder(nn.Module):
    def __init__(self, backbone_name: str = config.IMAGE_BACKBONE, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.out_dim = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = config.TEXT_MODEL):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.out_dim = self.backbone.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # mean pooling sobre tokens válidos
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts


class MultimodalKneeModel(nn.Module):
    def __init__(
        self,
        n_targets: int = len(config.TARGET_COLUMNS),
        image_backbone: str = config.IMAGE_BACKBONE,
        text_model: str = config.TEXT_MODEL,
        use_text: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_text = use_text
        self.image_encoder = ImageEncoder(image_backbone)
        fusion_dim = self.image_encoder.out_dim

        if use_text:
            self.text_encoder = TextEncoder(text_model)
            fusion_dim += self.text_encoder.out_dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_targets),
        )

    def forward(self, image, input_ids=None, attention_mask=None):
        img_feat = self.image_encoder(image)

        if self.use_text and input_ids is not None:
            text_feat = self.text_encoder(input_ids, attention_mask)
            feat = torch.cat([img_feat, text_feat], dim=1)
        else:
            feat = img_feat

        logits = self.head(feat)
        return logits  # aplicar sigmoid fora (BCEWithLogitsLoss cuida disso no treino)
