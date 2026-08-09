"""Testes de src/rsna_knee/modeling/heads.py."""

import torch

from src.rsna_knee.modeling.heads import build_mlp_head


def test_build_mlp_head_output_shape():
    head = build_mlp_head(feat_dim=128, n_targets=12, dropout=0.2, hidden_dim=256)
    x = torch.randn(4, 128)
    out = head(x)
    assert out.shape == (4, 12)


def test_build_mlp_head_eval_is_deterministic():
    head = build_mlp_head(feat_dim=16, n_targets=3, dropout=0.5, hidden_dim=32)
    head.eval()
    x = torch.randn(2, 16)
    out1 = head(x)
    out2 = head(x)
    assert torch.equal(out1, out2)


def test_build_mlp_head_gradient_flows():
    head = build_mlp_head(feat_dim=16, n_targets=3, dropout=0.0, hidden_dim=32)
    x = torch.randn(2, 16, requires_grad=True)
    loss = head(x).sum()
    loss.backward()
    grads = [p.grad for p in head.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)
