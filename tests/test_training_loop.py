"""Testes de src/rsna_knee/training/loop.py -- só as funções de metadados
de checkpoint (o loop de treino em si precisa de dados/GPU reais, fora do
escopo de testes unitários)."""

from src.rsna_knee.training.loop import (
    checkpoint_metadata_path,
    load_checkpoint_metadata,
    save_checkpoint_metadata,
)


def test_checkpoint_metadata_path_swaps_extension(tmp_path):
    ckpt = tmp_path / "best_fold0.pth"
    assert checkpoint_metadata_path(ckpt) == tmp_path / "best_fold0.json"


def test_save_and_load_checkpoint_metadata_roundtrip(tmp_path):
    ckpt = tmp_path / "best_fold0.pth"
    save_checkpoint_metadata(
        ckpt, fold=0, seed=42, val_auc=0.734, per_label_auc={"ACL": 0.8, "MCL": 0.6}
    )

    meta = load_checkpoint_metadata(ckpt)
    assert meta == {
        "fold": 0,
        "seed": 42,
        "val_auc": 0.734,
        "per_label_auc": {"ACL": 0.8, "MCL": 0.6},
    }


def test_load_checkpoint_metadata_missing_returns_none(tmp_path):
    ckpt = tmp_path / "best_fold0.pth"  # nunca salvo
    assert load_checkpoint_metadata(ckpt) is None
