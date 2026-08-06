"""
Gera submission.csv a partir de um checkpoint treinado.

Uso:
    python -m src.infer --checkpoint checkpoints/best_fold0.pth
"""

import argparse

# No Windows, importar torch depois de pandas pode quebrar o carregamento de
# DLL do torch (c10.dll) por conflito de dependências nativas -- torch
# precisa vir primeiro.
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from . import config
from .dataset import KneeDataset
from .model import MultimodalKneeModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Caminho do .pth treinado")
    parser.add_argument("--out", default=str(config.SUBMISSIONS_DIR / "submission.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ATENÇÃO: test.csv só tem StudyInstanceUID (sem coluna Report) -- o
    # texto do laudo não está disponível no teste. Se o modelo treinado
    # depende fortemente do texto, isso é um mismatch treino/teste real.
    # Ver nota em CLAUDE.md.
    test_df = pd.read_csv(config.TEST_CSV)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    test_ds = KneeDataset(
        test_df, config.TEST_SERIES_DIR, tokenizer, is_train=False,
        series_csv=config.TEST_SERIES_CSV,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = MultimodalKneeModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    study_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(image, input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()

            study_ids.extend(batch["study_id"])
            all_preds.append(preds)

    preds = np.concatenate(all_preds, axis=0)

    sub = pd.DataFrame(preds, columns=config.TARGET_COLUMNS)
    sub.insert(0, config.ID_COLUMN, study_ids)

    config.SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)
    sub.to_csv(args.out, index=False)
    print(f"Submissão salva em {args.out}")
    print(sub.head())


if __name__ == "__main__":
    main()
