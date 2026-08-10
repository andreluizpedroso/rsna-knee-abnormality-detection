"""TTA compatível com o pipeline Pilkwang v15 + patch da comunidade."""

from __future__ import annotations

import numpy as np

from . import constants


def window_starts(n_slice: int, group: int = constants.GROUP, overlap: bool = True) -> list[int]:
    """Inícios de janelas TTA sobre o eixo de slices."""
    if n_slice <= 0:
        return []
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    return [g * group for g in range(max(n_slice // group, 1))]


def aggregate_windows(
    preds_by_window: np.ndarray,
    target_pool: dict[str, str] | None = None,
    targets: list[str] | None = None,
) -> np.ndarray:
    """Agrega predições `(n_windows, n_targets)` por target.

    O default é média. Targets presentes em `target_pool` com valor `"max"`
    usam máximo, reproduzindo o patch TTA que elevou o score esperado para
    ~0.893 em T4.
    """
    arr = np.asarray(preds_by_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("preds_by_window precisa ter shape (n_windows, n_targets)")
    if arr.shape[0] == 0:
        raise ValueError("não há janelas TTA para agregar")

    targets = constants.TARGETS if targets is None else targets
    if arr.shape[1] != len(targets):
        raise ValueError(f"{arr.shape[1]} targets nas predições, esperado {len(targets)}")

    target_pool = constants.TTA_TARGET_POOL if target_pool is None else target_pool
    out = arr.mean(axis=0)
    for target, mode in target_pool.items():
        if target not in targets:
            continue
        idx = targets.index(target)
        if mode == "max":
            out[idx] = arr[:, idx].max()
        elif mode == "mean":
            out[idx] = arr[:, idx].mean()
        else:
            raise ValueError(f"modo TTA desconhecido para {target}: {mode}")
    return out

