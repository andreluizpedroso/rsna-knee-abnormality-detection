"""Cabeça de classificação aplicada sobre o embedding do backbone."""

import torch.nn as nn


def build_mlp_head(feat_dim: int, n_targets: int, dropout: float, hidden_dim: int) -> nn.Module:
    """MLP de 2 camadas (Dropout->Linear->ReLU->Dropout->Linear) que mapeia
    o embedding do backbone (`feat_dim`) pros `n_targets` logits de saída
    (sigmoid é aplicado fora -- `BCEWithLogitsLoss` cuida disso no
    treino)."""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(feat_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, n_targets),
    )
