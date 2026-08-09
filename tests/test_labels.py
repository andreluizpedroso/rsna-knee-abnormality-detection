"""Testes de src/rsna_knee/data/labels.py -- label_confidence_weights com
`evaluate_against_gold` mockado (a extração de texto em si roda sobre
laudos reais, fora do escopo de teste unitário; aqui o foco é a lógica de
mapeamento precisão -> peso)."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.rsna_knee import config
from src.rsna_knee.data.labels import EXCLUDED_FROM_PSEUDO_LABELS, label_confidence_weights


def _fake_metrics(precisions: dict) -> pd.DataFrame:
    rows = [
        {"label": label, "precision": precisions.get(label, np.nan)}
        for label in config.TARGET_COLUMNS
    ]
    return pd.DataFrame(rows).set_index("label")


def _non_excluded_labels(n: int) -> list:
    return [l for l in config.TARGET_COLUMNS if l not in EXCLUDED_FROM_PSEUDO_LABELS][:n]


def test_label_confidence_weights_excludes_configured_labels():
    metrics = _fake_metrics({label: 0.8 for label in config.TARGET_COLUMNS})
    with patch("src.rsna_knee.data.labels.evaluate_against_gold", return_value=metrics):
        weights = label_confidence_weights(pd.DataFrame())

    for excluded in EXCLUDED_FROM_PSEUDO_LABELS:
        assert excluded not in weights
    for label in config.TARGET_COLUMNS:
        if label not in EXCLUDED_FROM_PSEUDO_LABELS:
            assert label in weights


def test_label_confidence_weights_higher_precision_gets_higher_weight():
    precise_label, imprecise_label = _non_excluded_labels(2)
    metrics = _fake_metrics({precise_label: 0.9, imprecise_label: 0.55})
    with patch("src.rsna_knee.data.labels.evaluate_against_gold", return_value=metrics):
        weights = label_confidence_weights(pd.DataFrame())

    assert weights[precise_label] > weights[imprecise_label]


def test_label_confidence_weights_clips_to_max_bound():
    (label,) = _non_excluded_labels(1)
    metrics = _fake_metrics({label: 0.99})
    with patch("src.rsna_knee.data.labels.evaluate_against_gold", return_value=metrics):
        weights = label_confidence_weights(pd.DataFrame(), min_weight=0.1, max_weight=0.9)

    assert weights[label] == 0.9


def test_label_confidence_weights_clips_to_min_bound():
    (label,) = _non_excluded_labels(1)
    metrics = _fake_metrics({label: 0.01})
    with patch("src.rsna_knee.data.labels.evaluate_against_gold", return_value=metrics):
        weights = label_confidence_weights(pd.DataFrame(), min_weight=0.1, max_weight=0.9)

    assert weights[label] == 0.1


def test_label_confidence_weights_nan_precision_gets_min_weight():
    (label,) = _non_excluded_labels(1)
    metrics = _fake_metrics({})  # tudo NaN -- nenhuma predição coberta
    with patch("src.rsna_knee.data.labels.evaluate_against_gold", return_value=metrics):
        weights = label_confidence_weights(pd.DataFrame(), min_weight=0.15)

    assert weights[label] == 0.15
