"""Testes de src/rsna_knee/modeling/backbone.py -- timm.create_model
mockado (sem baixar/instanciar pesos reais)."""

from unittest.mock import MagicMock, patch

from src.rsna_knee.modeling.backbone import build_backbone


def test_build_backbone_passes_dynamic_img_size_for_dinov2():
    fake_backbone = MagicMock()
    fake_backbone.num_features = 384
    with patch(
        "src.rsna_knee.modeling.backbone.timm.create_model", return_value=fake_backbone
    ) as mocked:
        backbone, feat_dim = build_backbone("vit_small_patch14_dinov2.lvd142m", pretrained=False)

    assert backbone is fake_backbone
    assert feat_dim == 384
    _, kwargs = mocked.call_args
    assert kwargs.get("dynamic_img_size") is True


def test_build_backbone_no_dynamic_img_size_for_non_dinov2():
    fake_backbone = MagicMock()
    fake_backbone.num_features = 2048
    with patch(
        "src.rsna_knee.modeling.backbone.timm.create_model", return_value=fake_backbone
    ) as mocked:
        build_backbone("resnet50", pretrained=False)

    _, kwargs = mocked.call_args
    assert "dynamic_img_size" not in kwargs
