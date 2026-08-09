"""Testes de src/rsna_knee/modeling/attention_pool.py."""

import torch

from src.rsna_knee.modeling.attention_pool import AttentionPool


def test_attention_pool_output_shape():
    pool = AttentionPool(feat_dim=8)
    x = torch.randn(4, 3, 8)  # batch=4, n_instances=3, feat_dim=8
    out = pool(x)
    assert out.shape == (4, 8)


def test_attention_pool_single_instance_is_identity():
    pool = AttentionPool(feat_dim=8)
    pool.eval()
    x = torch.randn(2, 1, 8)  # só 1 instância -- softmax de 1 score = peso 1.0
    out = pool(x)
    assert torch.allclose(out, x.squeeze(1), atol=1e-5)


def test_attention_pool_masked_instance_does_not_affect_output():
    pool = AttentionPool(feat_dim=4)
    pool.eval()
    x = torch.randn(1, 3, 4)
    mask_drop_last = torch.tensor([[1.0, 1.0, 0.0]])

    out_before = pool(x, mask_drop_last)

    x_changed = x.clone()
    x_changed[0, 2] += 100.0  # só mexe na instância mascarada
    out_after = pool(x_changed, mask_drop_last)

    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_attention_pool_masking_changes_result_vs_unmasked():
    pool = AttentionPool(feat_dim=4)
    pool.eval()
    x = torch.randn(1, 3, 4)

    out_all = pool(x, torch.ones(1, 3))
    out_dropped = pool(x, torch.tensor([[1.0, 1.0, 0.0]]))

    assert not torch.allclose(out_all, out_dropped, atol=1e-5)
