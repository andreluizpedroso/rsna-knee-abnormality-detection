"""Métrica de validação: AUC-ROC macro, a métrica oficial da competição."""

import numpy as np
from sklearn.metrics import roc_auc_score

from .. import config


def macro_auc(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, dict[str, float]]:
    """AUC macro + AUC por label. Labels sem as duas classes são ignorados no
    cálculo (comum em folds pequenos/desbalanceados). Se `weights` for
    passado (peso por elemento -- ver `data.dataset.attach_label_weights`),
    só entram no cálculo os elementos com peso > 0: isso exclui células
    zero-preenchidas de pseudo-labels ausentes/abstidas, que senão
    contaminariam a AUC com "negativos" falsos."""
    aucs: dict[str, float] = {}
    for i, col in enumerate(config.TARGET_COLUMNS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if weights is not None:
            mask = weights[:, i] > 0
            yt, yp = yt[mask], yp[mask]
        if len(np.unique(yt)) < 2:
            continue
        aucs[col] = roc_auc_score(yt, yp)
    macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return macro, aucs
