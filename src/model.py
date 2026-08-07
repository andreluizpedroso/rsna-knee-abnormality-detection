"""Modelo baseline image-only: classifica as 12 anormalidades a partir de 1
imagem por estudo (ver dataset.py -- seleção de série + slice central).

O texto do laudo (Report) só é usado como weak supervision para gerar
pseudo-labels (weak_supervision.py), nunca como input do modelo -- ele não
existe em test.csv, então um modelo que dependesse dele pra prever teria
mismatch treino/teste real (ver CLAUDE.md)."""

import timm
import torch.nn as nn

from . import config


class KneeModel(nn.Module):
    def __init__(
        self,
        n_targets: int = len(config.TARGET_COLUMNS),
        backbone_name: str = config.IMAGE_BACKBONE,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Backbones DINOv2 (ver config.py) são ViT com patch size 14 -- por
        # padrão o timm exige que o input bata exatamente com o tamanho de
        # pré-treino (518x518), então precisamos de dynamic_img_size=True
        # pra aceitar o IMAGE_SIZE do projeto (tem que ser múltiplo de 14 --
        # ver config.IMAGE_SIZE). Não afeta os backbones CNN (resnet etc.),
        # que já aceitam qualquer resolução.
        backbone_kwargs = {"dynamic_img_size": True} if "dinov2" in backbone_name else {}
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, **backbone_kwargs
        )
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_targets),
        )

    def forward(self, image):
        feat = self.backbone(image)
        return self.head(feat)  # aplicar sigmoid fora (BCEWithLogitsLoss cuida disso no treino)
