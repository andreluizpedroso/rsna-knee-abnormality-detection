"""Construção do encoder de imagem (backbone) via `timm`."""

import torch.nn as nn
import timm


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Cria o backbone `name` (via `timm.create_model`, sem head de
    classificação -- `num_classes=0`) e retorna `(backbone, feat_dim)`.

    Backbones DINOv2 são ViT com patch size 14 -- por padrão o timm exige
    que o input bata exatamente com o tamanho de pré-treino (518x518),
    então precisamos de `dynamic_img_size=True` pra aceitar
    `config.IMAGE_SIZE` (tem que ser múltiplo de 14). Não afeta os
    backbones CNN (resnet etc.), que já aceitam qualquer resolução."""
    backbone_kwargs = {"dynamic_img_size": True} if "dinov2" in name else {}
    backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, **backbone_kwargs)
    feat_dim = backbone.num_features
    return backbone, feat_dim
