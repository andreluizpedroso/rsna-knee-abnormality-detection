"""Testes de src/rsna_knee/modeling/model.py -- KneeMILModel, com
build_backbone mockado (sem instanciar pesos reais)."""

from unittest.mock import patch

import torch
import torch.nn as nn

from src.rsna_knee.modeling.model import KneeMILModel


class _FakeBackbone(nn.Module):
    """Backbone de teste: retorna a média dos pixels de cada imagem do
    batch achatado, repetida pra um feat_dim fixo -- só pra verificar que
    o reshape (batch*n_instances -> batch, n_instances, feat_dim) e o
    attention pooling downstream funcionam, sem precisar de pesos reais."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vals = x.mean(dim=(1, 2, 3))
        return vals.unsqueeze(1).repeat(1, self.feat_dim)


def _make_model(feat_dim: int, n_targets: int) -> KneeMILModel:
    with patch(
        "src.rsna_knee.modeling.model.build_backbone",
        return_value=(_FakeBackbone(feat_dim), feat_dim),
    ):
        return KneeMILModel(n_targets=n_targets, pretrained=False)


def test_kneemilmodel_forward_output_shape():
    model = _make_model(feat_dim=16, n_targets=5)
    batch, n_instances, h, w = 2, 3, 4, 4
    images = torch.randn(batch, n_instances, 3, h, w)
    mask = torch.ones(batch, n_instances)

    out = model(images, mask)
    assert out.shape == (batch, 5)


def test_kneemilmodel_masked_instance_does_not_affect_output():
    model = _make_model(feat_dim=8, n_targets=3)
    model.eval()

    images = torch.zeros(1, 2, 3, 4, 4)
    images[0, 0] = 1.0
    images[0, 1] = 999.0  # instância mascarada -- não deveria influenciar
    mask = torch.tensor([[1.0, 0.0]])

    out_a = model(images, mask)

    images_b = images.clone()
    images_b[0, 1] = -500.0  # muda só a instância mascarada
    out_b = model(images_b, mask)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_kneemilmodel_no_mask_runs_without_error():
    model = _make_model(feat_dim=8, n_targets=3)
    images = torch.randn(2, 2, 3, 4, 4)
    out = model(images, mask=None)
    assert out.shape == (2, 3)
