"""Testes de src/rsna_knee/data/dataset.py -- attach_label_weights e
KneeDataset.__getitem__ com load_study_image mockado (sem DICOM real em
disco)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.rsna_knee import config
from src.rsna_knee.data.dataset import (
    KneeDataset,
    KneeMILDataset,
    attach_label_weights,
    load_study_bag,
)


def _make_df_with_targets(values_first_row):
    """1 linha, todas as config.TARGET_COLUMNS, valores conforme
    `values_first_row` (dict parcial -- as ausentes ficam NaN)."""
    row = {config.ID_COLUMN: "study1"}
    for col in config.TARGET_COLUMNS:
        row[col] = values_first_row.get(col, np.nan)
    return pd.DataFrame([row])


def test_attach_label_weights_zeroes_out_nan_targets():
    df = _make_df_with_targets({config.TARGET_COLUMNS[0]: 1.0})
    out = attach_label_weights(df, weight=0.5)

    present_col = config.TARGET_COLUMNS[0]
    missing_col = config.TARGET_COLUMNS[1]

    assert out.loc[0, f"{present_col}{config.WEIGHT_COLUMN_SUFFIX}"] == 0.5
    assert out.loc[0, f"{missing_col}{config.WEIGHT_COLUMN_SUFFIX}"] == 0.0
    # NaN do próprio target vira 0.0 (peso 0 já anula no loss)
    assert out.loc[0, missing_col] == 0.0
    assert out.loc[0, present_col] == 1.0


def test_attach_label_weights_accepts_per_label_dict():
    col_a, col_b = config.TARGET_COLUMNS[0], config.TARGET_COLUMNS[1]
    df = _make_df_with_targets({col_a: 1.0, col_b: 0.0})
    out = attach_label_weights(df, weight={col_a: 0.9, col_b: 0.2})

    assert out.loc[0, f"{col_a}{config.WEIGHT_COLUMN_SUFFIX}"] == 0.9
    assert out.loc[0, f"{col_b}{config.WEIGHT_COLUMN_SUFFIX}"] == 0.2


def test_attach_label_weights_dict_missing_label_uses_default_1():
    col_a = config.TARGET_COLUMNS[0]
    df = _make_df_with_targets({col_a: 1.0})
    out = attach_label_weights(df, weight={})  # nenhum label no dict

    assert out.loc[0, f"{col_a}{config.WEIGHT_COLUMN_SUFFIX}"] == 1.0


def test_kneedataset_getitem_with_mocked_image_load(tmp_path):
    df = _make_df_with_targets({config.TARGET_COLUMNS[0]: 1.0})
    df = attach_label_weights(df, weight=1.0)

    fake_image = np.zeros((3, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)

    with patch(
        "src.rsna_knee.data.dataset.load_study_image", return_value=fake_image
    ) as mocked:
        ds = KneeDataset(df, series_dir=tmp_path, is_train=True)
        item = ds[0]
        assert mocked.called

    assert item["study_id"] == "study1"
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert item["targets"].shape == (len(config.TARGET_COLUMNS),)
    assert item["target_weights"].shape == (len(config.TARGET_COLUMNS),)


def _series_df_for_bag():
    return pd.DataFrame([
        {
            config.ID_COLUMN: "study1", config.SERIES_ID_COLUMN: "series_a",
            "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1,
        },
        {
            config.ID_COLUMN: "study1", config.SERIES_ID_COLUMN: "series_b",
            "Anatomical_Plane": "Coronal", "Fluid_Sensitive": 0,
        },
    ])


def test_load_study_bag_pads_up_to_max_series(tmp_path):
    for name in ["series_a", "series_b"]:
        d = tmp_path / name
        d.mkdir()
        (d / "s0.dcm").write_bytes(b"")

    fake_slice = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
    with patch("src.rsna_knee.data.dataset.load_dicom_slice", return_value=fake_slice):
        images, mask = load_study_bag(tmp_path, "study1", _series_df_for_bag(), max_series=3)

    assert images.shape == (3, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert mask.tolist() == [1.0, 1.0, 0.0]  # 2 séries reais + 1 padding


def test_load_study_bag_no_series_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_study_bag(tmp_path, "study1", series_df=None, max_series=3)


def test_kneemildataset_getitem_with_mocked_bag_load(tmp_path):
    df = _make_df_with_targets({config.TARGET_COLUMNS[0]: 1.0})
    df = attach_label_weights(df, weight=1.0)

    fake_images = np.zeros((3, 3, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
    fake_mask = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    with patch(
        "src.rsna_knee.data.dataset.load_study_bag", return_value=(fake_images, fake_mask)
    ) as mocked:
        ds = KneeMILDataset(df, series_dir=tmp_path, is_train=True, max_series=3)
        item = ds[0]
        assert mocked.called

    assert item["study_id"] == "study1"
    assert item["images"].shape == (3, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert item["mask"].tolist() == [1.0, 1.0, 0.0]
    assert item["targets"].shape == (len(config.TARGET_COLUMNS),)
    assert item["target_weights"].shape == (len(config.TARGET_COLUMNS),)
