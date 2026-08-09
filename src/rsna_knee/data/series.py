"""Seleção de série dentro de um estudo (um estudo tem múltiplas séries
DICOM -- planos anatômicos e sequências diferentes).

`select_preferred_series_id`/`pick_series_dir` escolhem 1 única série (a
mais informativa por um score simples) -- usadas pelo pipeline padrão
(`data/dataset.KneeDataset`). `select_series_subset`/`pick_series_dirs`
escolhem até N séries, pré-requisito pra seleção multi-série com agregação
por atenção (ver `modeling/attention_pool.py`, `data/dataset.KneeMILDataset`).
"""

from pathlib import Path

import pandas as pd

from .. import config


def select_preferred_series_id(study_id: str, series_df: pd.DataFrame) -> str | None:
    """Escolhe a SeriesInstanceUID mais informativa para um estudo, usando
    train_series.csv/test_series.csv. Prioridade: sagital + fluid-sensitive
    (melhor visualização de ligamento/menisco) > sagital > fluid-sensitive
    em qualquer plano > qualquer série disponível."""
    candidates = series_df[series_df[config.ID_COLUMN] == study_id]
    if candidates.empty:
        return None

    is_sagittal = candidates["Anatomical_Plane"] == "Sagittal"
    is_fluid_sensitive = candidates["Fluid_Sensitive"] == 1
    score = is_sagittal.astype(int) * 2 + is_fluid_sensitive.astype(int)

    best_idx = score.idxmax()
    return candidates.loc[best_idx, config.SERIES_ID_COLUMN]


def select_series_subset(
    study_id: str, series_df: pd.DataFrame, max_series: int = 3
) -> list[str]:
    """Escolhe até `max_series` séries mais informativas de um estudo, pelo
    mesmo score de `select_preferred_series_id` (sagital+fluid-sensitive >
    sagital > fluid-sensitive > qualquer uma), em vez de só a melhor --
    generaliza a seleção de série única pra um "saco" de instâncias (MIL,
    ver `modeling/attention_pool.py`). Lista vazia se o estudo não constar
    em `series_df`."""
    candidates = series_df[series_df[config.ID_COLUMN] == study_id].copy()
    if candidates.empty:
        return []

    is_sagittal = candidates["Anatomical_Plane"] == "Sagittal"
    is_fluid_sensitive = candidates["Fluid_Sensitive"] == 1
    candidates["_score"] = is_sagittal.astype(int) * 2 + is_fluid_sensitive.astype(int)
    candidates = candidates.sort_values("_score", ascending=False)
    return candidates[config.SERIES_ID_COLUMN].head(max_series).tolist()


def pick_series_dirs(
    study_dir: Path,
    study_id: str | None = None,
    series_df: pd.DataFrame | None = None,
    max_series: int = 3,
) -> list[Path]:
    """Escolhe até `max_series` pastas de série dentro de um estudo (ver
    `select_series_subset`). Séries escolhidas mas não baixadas
    localmente são puladas (não interrompem a seleção das demais). Se
    `series_df`/`study_id` não forem passados, ou nenhuma série escolhida
    estiver disponível, cai pra até `max_series` pastas em ordem
    alfabética -- mesmo espírito de fallback de `pick_series_dir`."""
    dirs: list[Path] = []
    if study_id is not None and series_df is not None:
        for series_id in select_series_subset(study_id, series_df, max_series=max_series):
            candidate = study_dir / str(series_id)
            if candidate.is_dir():
                dirs.append(candidate)

    if dirs:
        return dirs

    return sorted(p for p in study_dir.iterdir() if p.is_dir())[:max_series]


def lookup_anatomical_plane(series_id: str, series_df: pd.DataFrame | None) -> str | None:
    """Retorna o `Anatomical_Plane` (`"Sagittal"`/`"Coronal"`/`"Axial"`)
    de `series_id` em `series_df`, ou `None` se `series_df` for `None` ou
    a série não constar nele -- usado por `laterality.py` pra decidir se o
    espelhamento se aplica (ver `laterality.MIRROR_PLANES`)."""
    if series_df is None:
        return None
    match = series_df[series_df[config.SERIES_ID_COLUMN] == series_id]
    if match.empty:
        return None
    return match.iloc[0]["Anatomical_Plane"]


def pick_series_dir(
    study_dir: Path,
    study_id: str | None = None,
    series_df: pd.DataFrame | None = None,
) -> Path | None:
    """Escolhe uma série (pasta) dentro de um estudo. Se `series_df` (de
    train_series.csv/test_series.csv) e `study_id` forem passados, prioriza a
    série sagital fluid-sensitive (ver `select_preferred_series_id`). Caso
    contrário, ou se a série preferida não estiver na pasta baixada, cai de
    volta para a primeira série em ordem alfabética."""
    if study_id is not None and series_df is not None:
        preferred_id = select_preferred_series_id(study_id, series_df)
        if preferred_id is not None:
            preferred_dir = study_dir / str(preferred_id)
            if preferred_dir.is_dir():
                return preferred_dir

    series_dirs = sorted(p for p in study_dir.iterdir() if p.is_dir())
    return series_dirs[0] if series_dirs else None
