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
from .attention_pool import AttentionPool
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


class KneeMILModel(nn.Module):
    """Variante multi-instância (MIL) de `KneeModel`: em vez de 1 imagem
    por estudo, recebe um "saco" de N séries (ver
    `data/dataset.KneeMILDataset`) -- extrai o embedding de cada instância
    com o MESMO backbone (pesos compartilhados) e agrega via
    `AttentionPool` antes da head de classificação, em vez de escolher só
    a série mais informativa.

    Ainda não é o modelo padrão do pipeline (ver roadmap item 7 em
    CLAUDE.md -- maior esforço/risco, não validado empiricamente ainda);
    convive com `KneeModel` sem substituí-lo."""

    def __init__(
        self,
        n_targets: int = len(config.TARGET_COLUMNS),
        backbone_name: str = config.IMAGE_BACKBONE,
        pretrained: bool = True,
        dropout: float = config.HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained)
        self.attention_pool = AttentionPool(feat_dim)
        self.head = build_mlp_head(
            feat_dim, n_targets, dropout, hidden_dim=config.HEAD_HIDDEN_DIM
        )

    def forward(self, images: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """`images`: `(batch, n_instances, 3, H, W)`. `mask`:
        `(batch, n_instances)`, 1 = instância válida, 0 = padding (ver
        `AttentionPool`)."""
        b, n, c, h, w = images.shape
        flat = images.view(b * n, c, h, w)
        feats = self.backbone(flat).view(b, n, -1)
        pooled = self.attention_pool(feats, mask)
        return self.head(pooled)
