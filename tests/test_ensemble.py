"""Testes de src/rsna_knee/modeling/ensemble.py."""

import json

import numpy as np
import pytest

from src.rsna_knee.modeling.ensemble import (
    rank_mean_ensemble,
    weighted_ensemble,
    weights_from_checkpoint_metadata,
    weights_from_val_auc,
)


def test_rank_mean_ensemble_invariant_to_calibration_when_ranking_agrees():
    # Modelo A e B discordam em escala mas concordam na ordenação relativa
    # dos 3 estudos -- rank-mean deve produzir o mesmo ranking que
    # qualquer um dos dois isoladamente.
    model_a = np.array([[0.1], [0.4], [0.9]])
    model_b = np.array([[0.3], [0.35], [0.99]])

    combined = rank_mean_ensemble([model_a, model_b])
    assert np.argsort(combined[:, 0]).tolist() == [0, 1, 2]


def test_rank_mean_ensemble_single_model_is_identity_ranking():
    model_a = np.array([[0.2], [0.8], [0.5]])
    combined = rank_mean_ensemble([model_a])
    expected = np.array([[1 / 3], [3 / 3], [2 / 3]])
    assert np.allclose(combined, expected)


def test_rank_mean_ensemble_empty_list_raises():
    with pytest.raises(ValueError):
        rank_mean_ensemble([])


def test_weighted_ensemble_favors_higher_weight_checkpoint():
    strong = np.array([[0.9]])
    weak = np.array([[0.1]])
    combined = weighted_ensemble([strong, weak], weights=[0.9, 0.5])
    simple_mean = (strong + weak) / 2
    # Ponderado deve ficar mais perto do checkpoint forte que a media simples
    assert abs(combined[0, 0] - strong[0, 0]) < abs(simple_mean[0, 0] - strong[0, 0])


def test_weighted_ensemble_single_model_is_identity():
    preds = np.array([[0.3, 0.7]])
    combined = weighted_ensemble([preds], weights=[1.0])
    assert np.allclose(combined, preds)


def test_weighted_ensemble_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        weighted_ensemble([np.zeros((1, 1))], weights=[0.5, 0.5])


def test_weighted_ensemble_nonpositive_weight_sum_raises():
    with pytest.raises(ValueError):
        weighted_ensemble([np.zeros((1, 1)), np.zeros((1, 1))], weights=[0.0, 0.0])


def test_rank_mean_ensemble_with_weights_favors_higher_weight_model():
    # 2 estudos, ordenacoes opostas entre os 2 modelos -- com peso maior
    # pro modelo A, o resultado deve refletir a ordenacao de A, nao a
    # media 50/50 (que empataria).
    model_a = np.array([[0.1], [0.9]])  # estudo0 < estudo1
    model_b = np.array([[0.9], [0.1]])  # estudo0 > estudo1
    combined = rank_mean_ensemble([model_a, model_b], weights=[0.9, 0.1])
    assert combined[0, 0] < combined[1, 0]


def test_rank_mean_ensemble_no_weights_is_uniform():
    model_a = np.array([[0.1], [0.9]])
    combined_no_w = rank_mean_ensemble([model_a, model_a])
    combined_uniform_w = rank_mean_ensemble([model_a, model_a], weights=[0.5, 0.5])
    assert np.allclose(combined_no_w, combined_uniform_w)


def test_weights_from_val_auc_higher_auc_gets_higher_weight():
    weights = weights_from_val_auc([0.60, 0.85])
    assert weights[1] > weights[0]
    assert abs(sum(weights) - 1.0) < 1e-9


def test_weights_from_val_auc_equal_aucs_uniform_weights():
    weights = weights_from_val_auc([0.7, 0.7, 0.7])
    assert all(abs(w - 1 / 3) < 1e-9 for w in weights)


def test_weights_from_val_auc_empty_raises():
    with pytest.raises(ValueError):
        weights_from_val_auc([])


def test_weights_from_checkpoint_metadata_reads_sidecar_json(tmp_path):
    ckpt_a = tmp_path / "best_fold0.pth"
    ckpt_b = tmp_path / "best_fold1.pth"
    ckpt_a.with_suffix(".json").write_text(json.dumps({"val_auc": 0.60}))
    ckpt_b.with_suffix(".json").write_text(json.dumps({"val_auc": 0.85}))

    weights = weights_from_checkpoint_metadata([ckpt_a, ckpt_b])
    assert weights[1] > weights[0]


def test_weights_from_checkpoint_metadata_missing_sidecar_raises(tmp_path):
    ckpt = tmp_path / "best_fold0.pth"  # sem .json ao lado
    with pytest.raises(ValueError):
        weights_from_checkpoint_metadata([ckpt])
