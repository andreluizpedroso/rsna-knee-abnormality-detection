"""
Gera submission.csv a partir de um ou mais checkpoints treinados. Com mais
de um checkpoint (ex.: um por fold do k-fold), faz ensemble por média
simples das probabilidades de cada modelo -- mais robusto que depender de
um único fold.

Uso:
    python -m src.infer --checkpoint checkpoints/best_fold0.pth
    python -m src.infer --checkpoint checkpoints/best_fold0.pth checkpoints/best_fold1.pth checkpoints/best_fold2.pth
"""

import argparse

# No Windows, importar torch depois de pandas pode quebrar o carregamento de
# DLL do torch (c10.dll) por conflito de dependências nativas -- torch
# precisa vir primeiro.
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from . import config
from .dataset import KneeDataset
from .model import KneeModel


def main():
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

    # pretrained=False: os pesos vêm todos do checkpoint carregado logo
    # abaixo, então baixar o ImageNet pré-treinado do timm seria só
    # desperdício de rede -- e a submissão roda sem internet mesmo.
    models = []
    for ckpt_path in args.checkpoint:
        model = KneeModel(pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models.append(model)
    print(f"Ensemble de {len(models)} checkpoint(s): {args.checkpoint}")

    study_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)

            # Média simples das probabilidades de cada modelo -- ensemble
            # por fold reduz a variância de um único split pequeno (58
            # estudos gold).
            batch_preds = torch.stack(
                [torch.sigmoid(model(image)) for model in models], dim=0
            ).mean(dim=0)

            study_ids.extend(batch["study_id"])
            all_preds.append(batch_preds.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)

    sub = pd.DataFrame(preds, columns=config.TARGET_COLUMNS)
    sub.insert(0, config.ID_COLUMN, study_ids)

    config.SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)
    sub.to_csv(args.out, index=False)
    print(f"Submissão salva em {args.out}")
    print(sub.head())


if __name__ == "__main__":
    main()
