"""Testes de src/rsna_knee/data/dicom.py -- I/O de baixo nível, sem GPU e
sem dados reais (arrays sintéticos)."""

import numpy as np

from src.rsna_knee.data.dicom import normalize_intensity, resize_to


def test_normalize_intensity_scales_to_unit_range():
    img = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    out = normalize_intensity(img)
    assert out.min() == 0.0
    assert abs(out.max() - 1.0) < 1e-5


def test_normalize_intensity_constant_image_does_not_raise():
    img = np.full((4, 4), 7.0, dtype=np.float32)
    out = normalize_intensity(img)
    assert np.isfinite(out).all()


def test_resize_to_preserves_target_shape():
    img = np.random.rand(256, 300).astype(np.float32)
    out = resize_to(img, size=100)
    assert out.shape == (100, 100)
