"""Loop de treino/validação por época e por fold.

`train_one_fold` recebe um `TrainRunConfig` explícito (em vez do
`argparse.Namespace` usado antes em `src/train.py`) -- desacopla a lógica
de treino da CLI (ver `cli/train.py`, que constrói o `TrainRunConfig` a
partir dos argumentos de linha de comando).
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

# No Windows, importar torch depois de pandas pode quebrar o carregamento de
# DLL do torch (c10.dll) por conflito de dependências nativas -- torch
# precisa vir primeiro.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm

from .. import config
from ..data.dataset import KneeDataset, attach_label_weights
from ..modeling.model import KneeModel
from .validation import macro_auc


@dataclass
class TrainRunConfig:
    """Parâmetros de uma rodada de treino, construídos em `cli/train.py` a
    partir do argparse -- ver módulo docstring."""

    smoke_test: bool = False
    epochs: int = config.EPOCHS
    seed: int = config.SEED


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    desc: str = "",
) -> tuple[float, float, dict[str, float]]:
    """Roda 1 época (treino ou validação, conforme `train`). Retorna
    `(avg_loss, macro_auc, auc_por_label)`."""
    model.train(mode=train)
    total_loss = 0.0
    total_weight = 0.0
    all_targets, all_preds, all_weights = [], [], []

    # Barra de progresso por batch -- sem isso, uma época que demora (I/O de
    # DICOM lido do dataset montado no Kaggle pode ser bem mais lento que o
    # cache local) fica sem nenhum sinal de vida nos logs até terminar.
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        targets = batch["targets"].to(device)
        weights = batch["target_weights"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(image)
            # reduction='none' + peso por elemento -- pseudo-labels (weak
            # supervision) pesam config.PSEUDO_LABEL_WEIGHT, labels reais
            # pesam 1.0, e células sem sinal (weight 0) não contribuem.
            raw_loss = criterion(logits, targets)
            weight_sum = weights.sum().clamp(min=1e-6)
            loss = (raw_loss * weights).sum() / weight_sum

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * weight_sum.item()
        total_weight += weight_sum.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_weights.append(weights.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_weight
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    w = np.concatenate(all_weights)
    macro, per_label = macro_auc(y_true, y_pred, weights=w)
    return avg_loss, macro, per_label


def checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    """Caminho do sidecar `.json` de metadados de um checkpoint -- mesmo
    nome, extensão trocada."""
    return Path(checkpoint_path).with_suffix(".json")


def save_checkpoint_metadata(
    checkpoint_path: Path,
    fold: int,
    seed: int,
    val_auc: float,
    per_label_auc: dict[str, float],
) -> None:
    """Salva um `.json` ao lado do checkpoint com a métrica de validação
    que o gerou -- pré-requisito pra ensemble ponderado por holdout (ver
    `modeling.ensemble.weighted_ensemble`/`weights_from_val_auc`), já que
    hoje essa métrica só era impressa no console, não persistida."""
    metadata = {
        "fold": fold,
        "seed": seed,
        "val_auc": val_auc,
        "per_label_auc": per_label_auc,
    }
    checkpoint_metadata_path(checkpoint_path).write_text(json.dumps(metadata, indent=2))


def load_checkpoint_metadata(checkpoint_path: Path) -> dict | None:
    """Lê o sidecar de metadados de um checkpoint (ver
    `save_checkpoint_metadata`). `None` se não existir (ex.: checkpoint
    antigo, salvo antes desse metadado existir)."""
    meta_path = checkpoint_metadata_path(checkpoint_path)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def filter_studies_with_local_images(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só as linhas cujo estudo tem pasta baixada localmente em
    config.TRAIN_SERIES_DIR/<StudyInstanceUID>/. Usado por --smoke-test pra
    restringir a amostra mínima baixada (ver scripts/download_sample_images.py)
    -- funciona pra qualquer subconjunto parcialmente baixado, não só um
    conjunto fixo de estudos."""
    has_local_images = df[config.ID_COLUMN].apply(
        lambda sid: (config.TRAIN_SERIES_DIR / str(sid)).exists()
    )
    return df[has_local_images].reset_index(drop=True)


def build_folds(
    gold_df: pd.DataFrame, smoke_test: bool
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Decide como dividir `gold_df` em folds de treino/validação e retorna
    a lista de `(fold_index, train_gold_df, val_df)`.

    Cada linha de train.csv já é um StudyInstanceUID único (não há coluna
    de paciente), então fold direto não tem risco de vazamento. Usa
    `MultilabelStratifiedKFold` (em vez de KFold/StratifiedKFold comum) para
    manter a proporção das 12 classes em cada fold -- com só 58 exemplos e
    MCL tendo apenas 9 positivos, um split aleatório arrisca folds sem
    nenhum positivo de alguma classe. Com pouquíssimos exemplos (ex.:
    --smoke-test com só 4-5 estudos, abaixo de `config.MIN_STUDIES_FOR_KFOLD`)
    a estratificação não faz sentido -- usa um split manual simples (última
    linha vira validação) só pra exercitar o pipeline, com um único
    "fold"."""
    if smoke_test and len(gold_df) < config.MIN_STUDIES_FOR_KFOLD:
        train_gold_df = gold_df.iloc[:-1].reset_index(drop=True)
        val_df = gold_df.iloc[-1:].reset_index(drop=True)
        return [(0, train_gold_df, val_df)]

    mskf = MultilabelStratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )
    gold_df = gold_df.copy()
    gold_df["fold"] = -1
    y = gold_df[config.TARGET_COLUMNS].values
    for f, (_, val_idx) in enumerate(mskf.split(gold_df, y)):
        gold_df.loc[val_idx, "fold"] = f

    # Treina TODOS os folds (não só o fold 0) -- cada um vira um
    # checkpoint próprio (best_fold{f}.pth) e a inferência
    # (inference/submission.py) pode ensemblá-los (média das
    # probabilidades) em vez de depender de um único fold.
    folds = []
    for f in range(config.N_FOLDS):
        train_gold_df = gold_df[gold_df["fold"] != f].reset_index(drop=True)
        val_df_f = gold_df[gold_df["fold"] == f].reset_index(drop=True)
        folds.append((f, train_gold_df, val_df_f))
    return folds


def train_one_fold(
    fold: int,
    train_gold_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    run_config: TrainRunConfig,
    device: torch.device,
    pseudo_label_weight: float | dict[str, float] = config.PSEUDO_LABEL_WEIGHT,
) -> float:
    """Treina um fold até `run_config.epochs` épocas ou até
    `config.EARLY_STOPPING_PATIENCE` épocas seguidas sem melhora de val_auc
    (o que vier primeiro). Salva o melhor checkpoint do fold em
    checkpoints/best_fold{fold}.pth (ou `best_fold{fold}_seed{seed}.pth` se
    `run_config.seed` for diferente de `config.SEED` -- permite treinar o
    mesmo fold com seeds diferentes sem sobrescrever, pro ensemble
    multi-seed), com um `.json` de metadados ao lado (ver
    `save_checkpoint_metadata`). Retorna o melhor val_auc alcançado.

    `pseudo_label_weight` aceita um escalar (comportamento original,
    `config.PSEUDO_LABEL_WEIGHT` uniforme) ou um `dict[str, float]` com
    peso por label -- ver `data.labels.label_confidence_weights`."""
    train_df = pd.concat(
        [
            attach_label_weights(train_gold_df, weight=1.0),
            attach_label_weights(pseudo_df, weight=pseudo_label_weight),
        ],
        ignore_index=True,
    )
    val_df = attach_label_weights(val_df, weight=1.0)

    train_ds: TorchDataset = KneeDataset(
        train_df, config.TRAIN_SERIES_DIR, is_train=True,
        series_csv=config.TRAIN_SERIES_CSV,
    )
    val_ds: TorchDataset = KneeDataset(
        val_df, config.TRAIN_SERIES_DIR, is_train=True,
        series_csv=config.TRAIN_SERIES_CSV,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    model = KneeModel().to(device)
    # Backbones DINOv2 (ViT pré-treinado) precisam de LR bem menor pro
    # backbone que pra head recém-inicializada -- LR=config.LR uniforme
    # (calibrado pro resnet50) destruiu os pesos pré-treinados nas
    # primeiras batches: train_auc ficou travado em ~0.50 (ruído) por 10
    # épocas inteiras no primeiro treino real com DINOv2 (ver PROGRESS.md).
    # CNNs fine-tunam bem com uma LR única; ViT não.
    if "dinov2" in config.IMAGE_BACKBONE:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": config.DINOV2_BACKBONE_LR},
                {"params": model.head.parameters(), "lr": config.LR},
            ],
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
    # reduction='none': a agregação ponderada (peso por elemento) acontece
    # em run_epoch, não aqui.
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_auc = -1.0
    epochs_without_improvement = 0

    for epoch in range(run_config.epochs):
        train_loss, train_auc, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True,
            desc=f"fold {fold} epoch {epoch+1}/{run_config.epochs} [train]",
        )
        val_loss, val_auc, val_per_label = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False,
            desc=f"fold {fold} epoch {epoch+1}/{run_config.epochs} [val]",
        )

        print(
            f"[fold {fold} epoch {epoch+1}/{run_config.epochs}] "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} | "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )
        print(f"  AUC por label (val): {val_per_label}")

        # val_auc pode ser nan (fold sem as duas classes de nenhum label) --
        # a comparação com nan é sempre False, então nunca conta como
        # melhora, o que é o comportamento certo.
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            config.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
            seed_suffix = "" if run_config.seed == config.SEED else f"_seed{run_config.seed}"
            ckpt_path = config.CHECKPOINT_DIR / f"best_fold{fold}{seed_suffix}.pth"
            torch.save(model.state_dict(), ckpt_path)
            save_checkpoint_metadata(ckpt_path, fold, run_config.seed, val_auc, val_per_label)
            print(f"  -> novo melhor modelo salvo (val_auc={val_auc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"  -> early stopping (sem melhora por "
                    f"{config.EARLY_STOPPING_PATIENCE} épocas)"
                )
                break

    print(f"Melhor val_auc (fold {fold}): {best_auc:.4f}")
    return best_auc
