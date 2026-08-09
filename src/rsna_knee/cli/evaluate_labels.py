"""
Avalia as regras de weak supervision (`data/labels.py`) contra os estudos
com label real e reporta a cobertura de pseudo-labels geradas.

Uso:
    python -m src.rsna_knee.cli.evaluate_labels
"""

import pandas as pd

from .. import config
from ..data.labels import EXCLUDED_FROM_PSEUDO_LABELS, evaluate_against_gold, generate_pseudo_labels


def main() -> None:
    df = pd.read_csv(config.TRAIN_CSV)

    print("=== Avaliação das regras contra os estudos com label verdadeiro ===")
    metrics = evaluate_against_gold(df)
    print(metrics.round(3))

    print(f"\n(Excluídas das pseudo-labels: {sorted(EXCLUDED_FROM_PSEUDO_LABELS)} -- gold insuficiente pra calibrar)")

    print("\n=== Gerando pseudo-labels para os estudos sem label completo ===")
    out = generate_pseudo_labels(df)
    n_pseudo = out["is_pseudo_label"].sum()
    print(f"Estudos que ganharam ao menos 1 pseudo-label: {n_pseudo}")
    other_cols = [c for c in config.TARGET_COLUMNS if c not in EXCLUDED_FROM_PSEUDO_LABELS]
    still_missing = out.loc[out["is_pseudo_label"], other_cols].isna().any(axis=1).sum()
    print(f"Desses, ainda com NaN em alguma das 10 colunas não-excluídas: {still_missing}")


if __name__ == "__main__":
    main()
