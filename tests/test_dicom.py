"""Testes de src/rsna_knee/data/dicom.py -- I/O de baixo nível, sem GPU e
sem dados reais (arrays sintéticos)."""

import numpy as np
import pydicom

from src.rsna_knee.data.dicom import (
    crop_to_physical_fov,
    normalize_intensity,
    read_pixel_spacing,
    resize_to,
)


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


def test_read_pixel_spacing_returns_row_col_tuple():
    ds = pydicom.Dataset()
    ds.PixelSpacing = [0.5, 0.75]
    assert read_pixel_spacing(ds) == (0.5, 0.75)


def test_read_pixel_spacing_missing_tag_returns_none():
    ds = pydicom.Dataset()
    assert read_pixel_spacing(ds) is None


def test_crop_to_physical_fov_returns_expected_pixel_size():
    img = np.zeros((100, 100), dtype=np.float32)
    # 0.5 mm/pixel, FOV alvo 40mm -> 40/0.5 = 80 pixels
    out = crop_to_physical_fov(img, pixel_spacing=(0.5, 0.5), fov_mm=40.0)
    assert out.shape == (80, 80)


def test_crop_to_physical_fov_is_centered():
    img = np.zeros((10, 10), dtype=np.float32)
    img[5, 5] = 1.0  # marcador no centro exato
    out = crop_to_physical_fov(img, pixel_spacing=(1.0, 1.0), fov_mm=4.0)
    assert out.shape == (4, 4)
    # top = left = (10-4)//2 = 3 -> o marcador em [5,5] cai em [5-3, 5-3]
    assert out[2, 2] == 1.0
    assert out.sum() == 1.0  # só o marcador, resto continua zero


def test_crop_to_physical_fov_pads_when_target_larger_than_image():
    img = np.ones((10, 10), dtype=np.float32) * 3.0
    # FOV alvo (20mm / 1.0 mm/px = 20px) maior que a imagem (10px) -- não
    # deve lançar exceção, deve fazer padding.
    out = crop_to_physical_fov(img, pixel_spacing=(1.0, 1.0), fov_mm=20.0)
    assert out.shape == (20, 20)
    assert np.all(out == 3.0)  # padding "edge" replica o valor constante
