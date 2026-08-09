"""Testes de src/rsna_knee/data/series.py -- DataFrames sintéticos em
memória, sem dados reais do Kaggle."""

import pandas as pd

from src.rsna_knee.data.series import pick_series_dir, select_preferred_series_id


def _series_df(rows):
    return pd.DataFrame(rows, columns=[
        "StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive",
    ])


def test_select_preferred_series_id_picks_sagittal_fluid_sensitive():
    df = _series_df([
        ("study1", "series_none", "Coronal", 0),
        ("study1", "series_fluid_only", "Coronal", 1),
        ("study1", "series_sagittal_only", "Sagittal", 0),
        ("study1", "series_sagittal_fluid", "Sagittal", 1),
    ])
    assert select_preferred_series_id("study1", df) == "series_sagittal_fluid"


def test_select_preferred_series_id_missing_study_returns_none():
    df = _series_df([("study1", "series_a", "Sagittal", 1)])
    assert select_preferred_series_id("study_not_present", df) is None


def test_pick_series_dir_falls_back_to_alphabetical(tmp_path):
    (tmp_path / "series_b").mkdir()
    (tmp_path / "series_a").mkdir()
    # sem series_df/study_id -- deve cair no fallback alfabético
    result = pick_series_dir(tmp_path)
    assert result == tmp_path / "series_a"


def test_pick_series_dir_falls_back_when_preferred_series_dir_missing(tmp_path):
    (tmp_path / "series_local_only").mkdir()
    df = _series_df([("study1", "series_not_downloaded", "Sagittal", 1)])
    result = pick_series_dir(tmp_path, study_id="study1", series_df=df)
    assert result == tmp_path / "series_local_only"
