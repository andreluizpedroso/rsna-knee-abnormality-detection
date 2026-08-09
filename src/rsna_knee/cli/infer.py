"""
Gera submission.csv a partir de um ou mais checkpoints treinados. Com mais
de um checkpoint (ex.: um por fold do k-fold), faz ensemble por média
simples das probabilidades de cada modelo -- mais robusto que depender de
um único fold.

Uso:
    python -m src.rsna_knee.cli.infer --checkpoint checkpoints/best_fold0.pth
    python -m src.rsna_knee.cli.infer --checkpoint checkpoints/best_fold0.pth checkpoints/best_fold1.pth checkpoints/best_fold2.pth
"""

import argparse

import torch
from torch.utils.data import DataLoader

import pandas as pd

from .. import config
from ..data.dataset import KneeDataset
from ..inference.submission import load_ensemble_models, predict_ensemble, write_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, nargs="+",
        help="Caminho de um ou mais .pth treinados (múltiplos -> ensemble por média)",
    )
    parser.add_argument("--out", default=str(config.SUBMISSIONS_DIR / "submission.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(config.TEST_CSV)
    test_ds = KneeDataset(
        test_df, config.TEST_SERIES_DIR, is_train=False,
        series_csv=config.TEST_SERIES_CSV,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    models = load_ensemble_models(args.checkpoint, device)
    print(f"Ensemble de {len(models)} checkpoint(s): {args.checkpoint}")

    study_ids, preds = predict_ensemble(models, test_loader, device)

    sub = write_submission(study_ids, preds, args.out)
    print(f"Submissão salva em {args.out}")
    print(sub.head())


if __name__ == "__main__":
    main()
