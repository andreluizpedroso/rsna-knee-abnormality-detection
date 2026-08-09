"""Testes de src/rsna_knee/data/series.py -- DataFrames sintéticos em
memória, sem dados reais do Kaggle."""

import pandas as pd

from src.rsna_knee.data.series import (
    lookup_anatomical_plane,
    pick_series_dir,
    pick_series_dirs,
    select_preferred_series_id,
    select_series_subset,
)


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


def test_lookup_anatomical_plane_found():
    df = _series_df([("study1", "series_a", "Coronal", 1)])
    assert lookup_anatomical_plane("series_a", df) == "Coronal"


def test_lookup_anatomical_plane_missing_series_returns_none():
    df = _series_df([("study1", "series_a", "Coronal", 1)])
    assert lookup_anatomical_plane("series_not_present", df) is None


def test_lookup_anatomical_plane_none_series_df_returns_none():
    assert lookup_anatomical_plane("series_a", None) is None


def test_select_series_subset_orders_by_score_and_limits():
    df = _series_df([
        ("study1", "series_none", "Coronal", 0),
        ("study1", "series_fluid_only", "Coronal", 1),
        ("study1", "series_sagittal_only", "Sagittal", 0),
        ("study1", "series_sagittal_fluid", "Sagittal", 1),
    ])
    result = select_series_subset("study1", df, max_series=2)
    assert result == ["series_sagittal_fluid", "series_sagittal_only"]


def test_select_series_subset_missing_study_returns_empty_list():
    df = _series_df([("study1", "series_a", "Sagittal", 1)])
    assert select_series_subset("study_not_present", df) == []


def test_pick_series_dirs_prioritizes_selected_series(tmp_path):
    for name in ["series_low", "series_high"]:
        (tmp_path / name).mkdir()
    df = _series_df([
        ("study1", "series_low", "Coronal", 0),
        ("study1", "series_high", "Sagittal", 1),
    ])
    result = pick_series_dirs(tmp_path, study_id="study1", series_df=df, max_series=2)
    assert result == [tmp_path / "series_high", tmp_path / "series_low"]


def test_pick_series_dirs_skips_series_not_downloaded_locally(tmp_path):
    (tmp_path / "series_local").mkdir()
    df = _series_df([
        ("study1", "series_not_local", "Sagittal", 1),  # não existe localmente
        ("study1", "series_local", "Coronal", 0),
    ])
    result = pick_series_dirs(tmp_path, study_id="study1", series_df=df, max_series=2)
    assert result == [tmp_path / "series_local"]


def test_pick_series_dirs_falls_back_to_alphabetical(tmp_path):
    for name in ["series_b", "series_a", "series_c"]:
        (tmp_path / name).mkdir()
    result = pick_series_dirs(tmp_path, max_series=2)
    assert result == [tmp_path / "series_a", tmp_path / "series_b"]
