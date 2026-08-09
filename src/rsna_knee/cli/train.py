"""
Loop de treino baseline com validação k-fold e AUC-ROC macro (a métrica oficial
da competição). Rode primeiro localmente numa amostra pequena para validar o
pipeline end-to-end antes de treinar full-scale num Kaggle Notebook com GPU.

Uso:
    python -m src.rsna_knee.cli.train
"""

import argparse

import torch
import numpy as np
import pandas as pd

from .. import config
from ..data.labels import generate_pseudo_labels
from ..training.loop import (
    TrainRunConfig,
    build_folds,
    filter_studies_with_local_images,
    set_seed,
    train_one_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Restringe treino/validação aos estudos com imagens já baixadas "
             "localmente (ver scripts/download_sample_images.py) -- só pra "
             "confirmar que o pipeline roda ponta a ponta sem erro, não "
             "produz um modelo útil.",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help="Override de config.EPOCHS (útil pro --smoke-test rodar rápido).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = TrainRunConfig(smoke_test=args.smoke_test, epochs=args.epochs)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(config.TRAIN_CSV)

    # gold_df: os 58 estudos com as 12 colunas de label REAIS preenchidas --
    # são a única fonte de verdade pra validação. O fold (K-fold) é montado
    # só sobre eles: cada um passa por validação em exatamente 1 dos folds,
    # nunca fica só no treino ao longo do processo.
    gold_df = df.dropna(subset=config.TARGET_COLUMNS).reset_index(drop=True)
    print(f"Estudos com label real: {len(gold_df)} / {len(df)}")

    # pseudo_df: estudos SEM label real que ganharam ao menos 1 pseudo-label
    # via weak supervision (Report -- ver data/labels.py). Entram SÓ no
    # treino, nunca na validação -- a validação precisa ficar limpa (só
    # gold) porque a precisão das regras (~0.5-0.8 contra o gold set) não é
    # confiável o bastante pra medir o modelo.
    pseudo_df = generate_pseudo_labels(df)
    pseudo_df = pseudo_df[pseudo_df["is_pseudo_label"]].reset_index(drop=True)
    print(f"Estudos com pseudo-label (só treino, peso {config.PSEUDO_LABEL_WEIGHT}): {len(pseudo_df)}")

    if run_config.smoke_test:
        gold_df = filter_studies_with_local_images(gold_df)
        pseudo_df = filter_studies_with_local_images(pseudo_df)
        print(f"[smoke-test] restringindo a estudos com imagem local: "
              f"{len(gold_df)} gold, {len(pseudo_df)} pseudo")

    config.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

    folds = build_folds(gold_df, smoke_test=run_config.smoke_test)
    fold_aucs = [
        train_one_fold(f, train_gold_df, val_df, pseudo_df, run_config, device)
        for f, train_gold_df, val_df in folds
    ]

    valid_aucs = [a for a in fold_aucs if not np.isnan(a) and a > -1.0]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    print(f"\nVal AUC por fold: {fold_aucs}")
    print(f"Val AUC médio entre folds: {mean_auc:.4f}")


if __name__ == "__main__":
    main()
