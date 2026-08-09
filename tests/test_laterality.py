"""Testes de src/rsna_knee/data/laterality.py -- usando pydicom.Dataset
sintético em memória (sem arquivo .dcm real em disco)."""

import numpy as np
import pydicom

from src.rsna_knee.config import CANONICAL_LATERALITY
from src.rsna_knee.data.laterality import apply_laterality_normalization, compute_laterality


def _make_dataset_with_geometry(ipp_x: float, laterality: str | None = None) -> pydicom.Dataset:
    ds = pydicom.Dataset()
    ds.ImagePositionPatient = [ipp_x, 0.0, 0.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.Rows = 10
    ds.Columns = 10
    ds.PixelSpacing = [1.0, 1.0]
    if laterality is not None:
        ds.Laterality = laterality
    return ds


def test_compute_laterality_positive_center_x_is_left():
    # center_x = ipp_x + (cols/2)*px_spacing[1] = ipp_x + 5
    ds = _make_dataset_with_geometry(ipp_x=10.0)
    assert compute_laterality(ds) == "L"


def test_compute_laterality_negative_center_x_is_right():
    ds = _make_dataset_with_geometry(ipp_x=-20.0)
    assert compute_laterality(ds) == "R"


def test_compute_laterality_falls_back_to_tag_when_geometry_absent():
    ds = pydicom.Dataset()
    ds.Laterality = "L"
    assert compute_laterality(ds) == "L"


def test_compute_laterality_abstains_when_nothing_available():
    ds = pydicom.Dataset()
    assert compute_laterality(ds) is None


def test_apply_laterality_normalization_mirrors_when_side_differs():
    img = np.zeros((4, 4), dtype=np.float32)
    img[0, 0] = 1.0  # marcador assimétrico no canto superior-esquerdo

    opposite_side = "L" if CANONICAL_LATERALITY == "R" else "R"
    ds = _make_dataset_with_geometry(
        ipp_x=10.0 if opposite_side == "L" else -20.0
    )
    assert compute_laterality(ds) == opposite_side

    out = apply_laterality_normalization(img, ds)
    assert np.array_equal(out, img[:, ::-1])


def test_apply_laterality_normalization_unchanged_when_side_matches_canonical():
    img = np.zeros((4, 4), dtype=np.float32)
    img[0, 0] = 1.0

    ds = _make_dataset_with_geometry(
        ipp_x=10.0 if CANONICAL_LATERALITY == "L" else -20.0
    )
    assert compute_laterality(ds) == CANONICAL_LATERALITY

    out = apply_laterality_normalization(img, ds)
    assert np.array_equal(out, img)
