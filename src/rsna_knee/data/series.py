"""Seleção de série dentro de um estudo (um estudo tem múltiplas séries
DICOM -- planos anatômicos e sequências diferentes).

Roadmap (ver CLAUDE.md/PROGRESS.md): hoje escolhe 1 única série por estudo
(a mais informativa por um score simples); seleção multi-série com
agregação por atenção é uma técnica futura que estende este módulo, não uma
reescrita dele.
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
