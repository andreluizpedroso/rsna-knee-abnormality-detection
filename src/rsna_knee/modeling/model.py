"""Modelo baseline image-only: classifica as 12 anormalidades a partir de 1
imagem por estudo (ver `data/dataset.py` -- seleção de série + slice
central). Composição fina de `backbone.py` + `heads.py`.

O texto do laudo (Report) só é usado como weak supervision para gerar
pseudo-labels (`data/labels.py`), nunca como input do modelo -- ele não
existe em test.csv, então um modelo que dependesse dele pra prever teria
mismatch treino/teste real (ver CLAUDE.md)."""

import torch
import torch.nn as nn

from .. import config
from .backbone import build_backbone
from .heads import build_mlp_head


class KneeModel(nn.Module):
    def __init__(
        self,
        n_targets: int = len(config.TARGET_COLUMNS),
        backbone_name: str = config.IMAGE_BACKBONE,
        pretrained: bool = True,
        dropout: float = config.HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained)
        self.head = build_mlp_head(
            feat_dim, n_targets, dropout, hidden_dim=config.HEAD_HIDDEN_DIM
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image)
        return self.head(feat)  # aplicar sigmoid fora (BCEWithLogitsLoss cuida disso no treino)
