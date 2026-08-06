"""
Loop de treino baseline com validação k-fold e AUC-ROC macro (a métrica oficial
da competição). Rode primeiro localmente numa amostra pequena para validar o
pipeline end-to-end antes de treinar full-scale num Kaggle Notebook com GPU.

Uso:
    python -m src.train
"""

import argparse
import random

# No Windows, importar torch depois de pandas pode quebrar o carregamento de
# DLL do torch (c10.dll) por conflito de dependências nativas -- torch
# precisa vir primeiro.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from . import config
from .dataset import KneeDataset, attach_label_weights
from .model import MultimodalKneeModel
from .weak_supervision import generate_pseudo_labels


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_auc(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, dict]:
    """AUC macro + AUC por label. Labels sem as duas classes são ignorados no
    cálculo (comum em folds pequenos/desbalanceados). Se `weights` for
    passado (peso por elemento -- ver dataset.attach_label_weights), só
    entram no cálculo os elementos com peso > 0: isso exclui células
    zero-preenchidas de pseudo-labels ausentes/abstidas, que senão
    contaminariam a AUC com "negativos" falsos."""
    aucs = {}
    for i, col in enumerate(config.TARGET_COLUMNS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if weights is not None:
            mask = weights[:, i] > 0
            yt, yp = yt[mask], yp[mask]
        if len(np.unique(yt)) < 2:
            continue
        aucs[col] = roc_auc_score(yt, yp)
    macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return macro, aucs


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    total_weight = 0.0
    all_targets, all_preds, all_weights = [], [], []

    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        weights = batch["target_weights"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(image, input_ids, attention_mask)
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

    avg_loss = total_loss / total_weight
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    w = np.concatenate(all_weights)
    macro, per_label = macro_auc(y_true, y_pred, weights=w)
    return avg_loss, macro, per_label


def _filter_studies_with_local_images(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só as linhas cujo estudo tem pasta baixada localmente em
    config.TRAIN_SERIES_DIR/<StudyInstanceUID>/. Usado por --smoke-test pra
    restringir a amostra mínima baixada (ver scripts/download_sample_images.py)
    -- funciona pra qualquer subconjunto parcialmente baixado, não só um
    conjunto fixo de estudos."""
    has_local_images = df[config.ID_COLUMN].apply(
        lambda sid: (config.TRAIN_SERIES_DIR / str(sid)).exists()
    )
    return df[has_local_images].reset_index(drop=True)


def parse_args():
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


def main():
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(config.TRAIN_CSV)

    # gold_df: os 58 estudos com as 12 colunas de label REAIS preenchidas --
    # são a única fonte de verdade pra validação. O fold (K-fold) é montado
    # só sobre eles: cada um passa por validação em exatamente 1 dos 3 folds,
    # nunca fica só no treino ao longo do processo.
    gold_df = df.dropna(subset=config.TARGET_COLUMNS).reset_index(drop=True)
    print(f"Estudos com label real: {len(gold_df)} / {len(df)}")

    # pseudo_df: estudos SEM label real que ganharam ao menos 1 pseudo-label
    # via weak supervision (Report -- ver src/weak_supervision.py). Entram
    # SÓ no treino, nunca na validação -- a validação precisa ficar limpa
    # (só gold) porque a precisão das regras (~0.5-0.8 contra o gold set) não
    # é confiável o bastante pra medir o modelo.
    pseudo_df = generate_pseudo_labels(df)
    pseudo_df = pseudo_df[pseudo_df["is_pseudo_label"]].reset_index(drop=True)
    print(f"Estudos com pseudo-label (só treino, peso {config.PSEUDO_LABEL_WEIGHT}): {len(pseudo_df)}")

    if args.smoke_test:
        gold_df = _filter_studies_with_local_images(gold_df)
        pseudo_df = _filter_studies_with_local_images(pseudo_df)
        print(f"[smoke-test] restringindo a estudos com imagem local: "
              f"{len(gold_df)} gold, {len(pseudo_df)} pseudo")

    # Cada linha de train.csv já é um StudyInstanceUID único (não há coluna
    # de paciente), então fold direto não tem risco de vazamento. Usa
    # MultilabelStratifiedKFold (em vez de KFold/StratifiedKFold comum) para
    # manter a proporção das 12 classes em cada fold -- com só 58 exemplos e
    # MCL tendo apenas 9 positivos, um split aleatório arrisca folds sem
    # nenhum positivo de alguma classe. Com pouquíssimos exemplos (ex.:
    # --smoke-test com só 4-5 estudos) a estratificação não faz sentido --
    # usa um split manual simples (última linha vira validação) só pra
    # exercitar o pipeline.
    fold = 0  # treinar só o fold 0 no baseline; loop pelos folds depois
    if args.smoke_test and len(gold_df) < config.N_FOLDS * 2:
        train_gold_df = gold_df.iloc[:-1].reset_index(drop=True)
        val_df = gold_df.iloc[-1:].reset_index(drop=True)
    else:
        mskf = MultilabelStratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
        )
        gold_df["fold"] = -1
        y = gold_df[config.TARGET_COLUMNS].values
        for f, (_, val_idx) in enumerate(mskf.split(gold_df, y)):
            gold_df.loc[val_idx, "fold"] = f
        train_gold_df = gold_df[gold_df["fold"] != fold].reset_index(drop=True)
        val_df = gold_df[gold_df["fold"] == fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)

    train_df = pd.concat(
        [
            attach_label_weights(train_gold_df, weight=1.0),
            attach_label_weights(pseudo_df, weight=config.PSEUDO_LABEL_WEIGHT),
        ],
        ignore_index=True,
    )
    val_df = attach_label_weights(val_df, weight=1.0)

    train_ds = KneeDataset(
        train_df, config.TRAIN_SERIES_DIR, tokenizer, is_train=True,
        series_csv=config.TRAIN_SERIES_CSV,
    )
    val_ds = KneeDataset(
        val_df, config.TRAIN_SERIES_DIR, tokenizer, is_train=True,
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

    model = MultimodalKneeModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    # reduction='none': a agregação ponderada (peso por elemento) acontece
    # em run_epoch, não aqui.
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    config.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    best_auc = -1.0

    for epoch in range(args.epochs):
        train_loss, train_auc, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_loss, val_auc, val_per_label = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )

        print(
            f"[epoch {epoch+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} | "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )
        print(f"  AUC por label (val): {val_per_label}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / f"best_fold{fold}.pth")
            print(f"  -> novo melhor modelo salvo (val_auc={val_auc:.4f})")

    print(f"Melhor val_auc (fold {fold}): {best_auc:.4f}")


if __name__ == "__main__":
    main()
