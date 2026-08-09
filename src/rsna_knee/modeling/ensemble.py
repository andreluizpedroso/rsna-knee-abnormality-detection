"""Ensemble entre modelos/checkpoints (ex.: folds diferentes de um mesmo
treino, ou seeds diferentes).

`rank_mean_ensemble`/`weighted_ensemble` combinam predições já calculadas;
`weights_from_val_auc`/`weights_from_checkpoint_metadata` derivam os pesos
a partir da métrica de validação persistida ao lado de cada checkpoint
(ver `training.loop.save_checkpoint_metadata`) -- ensemble multi-seed/fold
ponderado por holdout.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def rank_mean_ensemble(
    preds_per_model: list[np.ndarray], weights: list[float] | None = None
) -> np.ndarray:
    """Combina as predições de múltiplos modelos convertendo cada uma em
    RANKS percentuais por coluna (target) antes de agregar entre modelos,
    em vez de média direta de probabilidade.

    Cada elemento de `preds_per_model` é um array `(n_studies, n_targets)`
    das probabilidades de 1 modelo pra todos os estudos. Rank-mean
    neutraliza diferenças de calibração de probabilidade entre modelos
    treinados separadamente (comum quando os folds têm poucos exemplos,
    como aqui com 58 gold) -- mantém só a informação de ORDENAÇÃO relativa
    entre estudos, que é o que a métrica AUC de fato mede.

    Se `weights` for passado, agrega os ranks por MÉDIA PONDERADA (ver
    `weights_from_val_auc`) em vez de média simples; sem `weights`, todo
    modelo pesa igual."""
    if not preds_per_model:
        raise ValueError("rank_mean_ensemble precisa de pelo menos 1 conjunto de predições")
    ranked = [pd.DataFrame(p).rank(pct=True).to_numpy() for p in preds_per_model]
    if weights is None:
        return np.mean(ranked, axis=0)
    return weighted_ensemble(ranked, weights)


def weighted_ensemble(preds_per_model: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Média ponderada das predições (ou ranks) de múltiplos modelos --
    usada quando já se tem um peso por modelo (ex.: proporcional ao
    val_auc do fold/seed que o gerou, ver `weights_from_val_auc`). Pesos
    são normalizados internamente (não precisam somar 1)."""
    if not preds_per_model:
        raise ValueError("weighted_ensemble precisa de pelo menos 1 conjunto de predições")
    if len(preds_per_model) != len(weights):
        raise ValueError(
            f"{len(preds_per_model)} conjunto(s) de predições mas {len(weights)} peso(s)"
        )
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("soma dos pesos precisa ser positiva")
    w = w / w.sum()
    stacked = np.stack(preds_per_model, axis=0)  # (n_models, n_studies, n_targets)
    return np.tensordot(w, stacked, axes=(0, 0))


def weights_from_val_auc(val_aucs: list[float], temperature: float = 0.1) -> list[float]:
    """Converte 1 val_auc por checkpoint num peso relativo via softmax --
    acentua a diferença entre folds/seeds bons e ruins mais do que usar o
    val_auc bruto como peso direto, já que a escala de AUC costuma ficar
    comprimida perto de 1.0 (uma diferença de 0.02 ali é grande, não
    pequena). `temperature` mais baixa acentua mais a diferença entre os
    checkpoints; mais alta se aproxima de pesos uniformes."""
    if not val_aucs:
        raise ValueError("weights_from_val_auc precisa de pelo menos 1 val_auc")
    aucs = np.asarray(val_aucs, dtype=np.float64)
    scaled = aucs / temperature
    exp = np.exp(scaled - scaled.max())  # estabilidade numérica (evita overflow)
    return (exp / exp.sum()).tolist()


def weights_from_checkpoint_metadata(
    checkpoint_paths: list[Path], temperature: float = 0.1
) -> list[float]:
    """Lê o `.json` de metadados salvo ao lado de cada checkpoint (ver
    `training.loop.save_checkpoint_metadata`) e deriva pesos por
    `weights_from_val_auc`. Levanta `ValueError` se algum checkpoint não
    tiver metadados (ex.: checkpoint salvo antes desse recurso existir) --
    silenciosamente ponderar só uma parte do ensemble seria pior que
    recusar e deixar explícito que falta retreinar/gerar o sidecar."""
    from ..training.loop import load_checkpoint_metadata  # import local: evita import circular

    val_aucs = []
    for path in checkpoint_paths:
        meta = load_checkpoint_metadata(path)
        if meta is None or meta.get("val_auc") is None:
            raise ValueError(
                f"sem metadados de val_auc pra {path} -- gere de novo com a "
                f"versão atual de training/loop.py (salva um .json ao lado "
                f"do checkpoint)"
            )
        val_aucs.append(meta["val_auc"])
    return weights_from_val_auc(val_aucs, temperature=temperature)
