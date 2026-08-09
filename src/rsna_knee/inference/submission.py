"""Geração da submissão: carrega checkpoints, roda inferência em ensemble
(hoje: média simples de sigmoid) e escreve o `submission.csv` final.

Extraído do antigo `src/infer.py` (que tinha tudo dentro de um único
`main()`) em 3 funções nomeadas -- pré-requisito prático pro roadmap de TTA
e ensemble ponderado por holdout (ver `tta.py`, `modeling/ensemble.py`),
que vão reaproveitar `load_ensemble_models`/`write_submission` sem duplicar.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from .. import config
from ..modeling.model import KneeModel


def load_ensemble_models(checkpoint_paths: list[str], device: torch.device) -> list[KneeModel]:
    """Carrega um `KneeModel` por checkpoint em `checkpoint_paths`, em modo
    `eval()`. `pretrained=False`: os pesos vêm todos do checkpoint, então
    baixar o ImageNet pré-treinado do timm seria só desperdício de rede --
    e a submissão roda sem internet mesmo."""
    models = []
    for ckpt_path in checkpoint_paths:
        model = KneeModel(pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models.append(model)
    return models


def predict_ensemble(
    models: list[KneeModel], loader: DataLoader, device: torch.device
) -> tuple[list[str], np.ndarray]:
    """Roda inferência em `loader` com todos os `models` e agrega por média
    simples das probabilidades (sigmoid) de cada modelo -- ensemble por
    fold reduz a variância de um único split pequeno (58 estudos gold).
    Retorna `(study_ids, preds)`."""
    study_ids: list[str] = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)

            batch_preds = torch.stack(
                [torch.sigmoid(model(image)) for model in models], dim=0
            ).mean(dim=0)

            study_ids.extend(batch["study_id"])
            all_preds.append(batch_preds.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    return study_ids, preds


def write_submission(study_ids: list[str], preds: np.ndarray, out_path: Path) -> pd.DataFrame:
    """Monta o DataFrame de submissão (`config.ID_COLUMN` + 12 colunas de
    target) e grava em `out_path`. Retorna o DataFrame gravado."""
    sub = pd.DataFrame(preds, columns=config.TARGET_COLUMNS)
    sub.insert(0, config.ID_COLUMN, study_ids)

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    sub.to_csv(out_path, index=False)
    return sub
