"""
Gera submission.csv a partir de um ou mais checkpoints treinados. Com mais
de um checkpoint (ex.: um por fold do k-fold), faz ensemble entre modelos --
por padrão média simples de sigmoid; com `--tta`, roda TTA de slice
(vários slices vizinhos ao central) e ensemble por rank-mean, mais robusto
mas mais lento (ver `inference/tta.py`, `modeling/ensemble.py`). Com
`--weight-by-holdout` (requer `--tta` e checkpoints com `.json` de
metadados -- ver `training/loop.py`), pondera a contribuição de cada
checkpoint no ensemble proporcionalmente ao seu val_auc, em vez de peso
uniforme.

Uso:
    python -m src.rsna_knee.cli.infer --checkpoint checkpoints/best_fold0.pth
    python -m src.rsna_knee.cli.infer --checkpoint checkpoints/best_fold0.pth checkpoints/best_fold1.pth checkpoints/best_fold2.pth
    python -m src.rsna_knee.cli.infer --checkpoint checkpoints/best_fold0.pth --tta
    python -m src.rsna_knee.cli.infer --checkpoint checkpoints/best_fold0.pth checkpoints/best_fold1_seed7.pth --tta --weight-by-holdout
"""

import argparse

import torch
from torch.utils.data import DataLoader

import pandas as pd

from .. import config
from ..data.dataset import KneeDataset
from ..inference.submission import (
    load_ensemble_models,
    predict_ensemble,
    predict_ensemble_with_tta,
    write_submission,
)
from ..modeling.ensemble import weights_from_checkpoint_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, nargs="+",
        help="Caminho de um ou mais .pth treinados (múltiplos -> ensemble por média)",
    )
    parser.add_argument("--out", default=str(config.SUBMISSIONS_DIR / "submission.csv"))
    parser.add_argument(
        "--tta", action="store_true",
        help="TTA de slice + ensemble por rank-mean, em vez do caminho "
             "padrão (1 slice, média de sigmoid). Mais lento.",
    )
    parser.add_argument(
        "--tta-windows", type=int, default=config.TTA_WINDOWS,
        help="Nº de slices vizinhos ao centro usados no --tta.",
    )
    parser.add_argument(
        "--weight-by-holdout", action="store_true",
        help="Pondera o ensemble pelo val_auc de cada checkpoint (requer "
             "--tta e os .json de metadados salvos pelo treino atual -- "
             "ver training/loop.py). Sem isso, todo checkpoint pesa igual.",
    )
    args = parser.parse_args()

    if args.weight_by_holdout and not args.tta:
        raise SystemExit("--weight-by-holdout requer --tta (ensemble ponderado só existe no caminho de TTA)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(config.TEST_CSV)
    models = load_ensemble_models(args.checkpoint, device)
    print(f"Ensemble de {len(models)} checkpoint(s): {args.checkpoint}")

    if args.tta:
        weights = None
        if args.weight_by_holdout:
            weights = weights_from_checkpoint_metadata(args.checkpoint)
            print(f"Pesos por holdout: {dict(zip(args.checkpoint, weights))}")
        series_df = pd.read_csv(config.TEST_SERIES_CSV)
        study_ids = test_df[config.ID_COLUMN].tolist()
        study_ids, preds = predict_ensemble_with_tta(
            models, study_ids, config.TEST_SERIES_DIR, series_df, device,
            n_windows=args.tta_windows, weights=weights,
        )
    else:
        test_ds = KneeDataset(
            test_df, config.TEST_SERIES_DIR, is_train=False,
            series_csv=config.TEST_SERIES_CSV,
        )
        test_loader = DataLoader(
            test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS,
        )
        study_ids, preds = predict_ensemble(models, test_loader, device)

    sub = write_submission(study_ids, preds, args.out)
    print(f"Submissão salva em {args.out}")
    print(sub.head())


if __name__ == "__main__":
    main()
