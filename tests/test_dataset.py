"""Testes de src/rsna_knee/data/dataset.py -- attach_label_weights e
KneeDataset.__getitem__ com load_study_image mockado (sem DICOM real em
disco)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.rsna_knee import config
from src.rsna_knee.data.dataset import KneeDataset, attach_label_weights


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
