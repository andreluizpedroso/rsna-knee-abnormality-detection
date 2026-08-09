"""Geração da submissão: carrega checkpoints, roda inferência em ensemble
e escreve o `submission.csv` final.

Dois caminhos de predição:
- `predict_ensemble`: 1 slice por estudo (via `KneeDataset`/`DataLoader`),
  ensemble por média simples de sigmoid. Caminho original, mais rápido.
- `predict_ensemble_with_tta`: TTA de slice (`inference.tta`) + ensemble
  por rank-mean (`modeling.ensemble`) -- mais lento (N forwards por
  estudo em vez de 1), mas mais robusto a variância de qual slice é "o
  central" e a diferença de calibração entre checkpoints.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from .. import config
from ..modeling.ensemble import rank_mean_ensemble
from ..modeling.model import KneeModel
from .tta import predict_study_with_tta


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


def predict_ensemble_with_tta(
    models: list[KneeModel],
    study_ids: list[str],
    series_dir: Path,
    series_df: pd.DataFrame | None,
    device: torch.device,
    n_windows: int = config.TTA_WINDOWS,
    weights: list[float] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Prediz cada estudo em `study_ids` com TTA de slice
    (`inference.tta.predict_study_with_tta`) pra cada modelo, e agrega as
    predições entre modelos por rank-mean (`modeling.ensemble.rank_mean_ensemble`)
    em vez da média simples de sigmoid de `predict_ensemble`. `weights`
    (1 por modelo, na mesma ordem de `models`) pondera o ensemble --
    ver `modeling.ensemble.weights_from_checkpoint_metadata`; sem
    `weights`, todo modelo pesa igual. Retorna `(study_ids, preds)` na
    mesma ordem de `study_ids` (não da ordem de um `DataLoader`, já que
    aqui a leitura é feita 1 estudo por vez)."""
    per_model_preds = []
    for model in models:
        preds = [
            predict_study_with_tta(
                model, series_dir / str(study_id), study_id, series_df, device,
                n_windows=n_windows,
            )
            for study_id in study_ids
        ]
        per_model_preds.append(np.stack(preds, axis=0))

    preds = rank_mean_ensemble(per_model_preds, weights=weights)
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
