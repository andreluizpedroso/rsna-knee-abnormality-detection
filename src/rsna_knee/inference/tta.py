"""Test-time augmentation (TTA) por janelas de slice sobrepostas.

Hoje a inferência usa 1 slice fixo por estudo (o do meio da série, já
ordenada geometricamente -- ver `data/slices.pick_middle_slice`). Pequenas
variações de qual slice é "o meio" (série com nº par de slices, off-by-one)
já mudam a predição. Rodar o forward em vários slices vizinhos ao central e
agregar as probabilidades reduz essa variância sem precisar re-treinar
nada.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .. import config
from ..data import series as series_mod
from ..data.laterality import load_dicom_slice
from ..data.slices import order_slices_by_geometry


def tta_slice_paths(series_dir: Path, n_windows: int = config.TTA_WINDOWS) -> list[Path]:
    """Escolhe até `n_windows` slices consecutivos centrados no meio da
    série (pela ordem geométrica real -- ver
    `data.slices.order_slices_by_geometry`), pra rodar TTA de slice na
    inferência em vez de depender de um único slice central. Se a série
    tiver menos slices que `n_windows`, retorna todos os disponíveis."""
    ordered = order_slices_by_geometry(series_dir)
    n = len(ordered)
    if n == 0:
        return []

    mid = n // 2
    half = n_windows // 2
    start = max(0, mid - half)
    end = min(n, start + n_windows)
    start = max(0, end - n_windows)  # reajusta se o fim bateu no limite da série
    return ordered[start:end]


@torch.no_grad()
def predict_study_with_tta(
    model: nn.Module,
    study_dir: Path,
    study_id: str | None,
    series_df: pd.DataFrame | None,
    device: torch.device,
    image_size: int = config.IMAGE_SIZE,
    n_windows: int = config.TTA_WINDOWS,
) -> np.ndarray:
    """Prediz as probabilidades de 1 estudo com TTA de slice: roda o
    forward do `model` em cada slice de `tta_slice_paths` (mesma série,
    lateralidade normalizada por slice) e retorna a MÉDIA das
    probabilidades (sigmoid) entre as janelas -- shape `(n_targets,)`."""
    series_dir = series_mod.pick_series_dir(study_dir, study_id, series_df)
    if series_dir is None:
        raise FileNotFoundError(f"Nenhuma série encontrada em {study_dir}")

    slice_paths = tta_slice_paths(series_dir, n_windows=n_windows)
    if not slice_paths:
        raise FileNotFoundError(f"Nenhum slice .dcm encontrado em {series_dir}")

    plane = series_mod.lookup_anatomical_plane(series_dir.name, series_df)
    imgs = [
        load_dicom_slice(p, image_size=image_size, plane=plane)
        for p in slice_paths
    ]
    # (n_windows, 3, H, W) -- 3 canais replicados, igual data.dataset.load_study_image
    batch = np.stack([np.stack([img] * 3, axis=0) for img in imgs], axis=0)
    tensor = torch.from_numpy(batch).to(device)

    was_training = model.training
    model.eval()
    logits = model(tensor)
    if was_training:
        model.train()

    return torch.sigmoid(logits).mean(dim=0).cpu().numpy()
