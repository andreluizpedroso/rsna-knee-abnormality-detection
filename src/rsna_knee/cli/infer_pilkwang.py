"""Inferência compatível com o pipeline público Pilkwang v15.

Esta CLI valida o pacote de pesos e o ambiente Kaggle antes de rodar a
submissão forte. A reprodução completa do score exige o dataset/modelos
anexados no Kaggle; localmente use `--dry-run` para validar manifest e
contratos sem carregar DICOMs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import config
from ..inference.submission import write_submission
from ..pilkwang.infer import infer_from_manifest
from ..pilkwang.manifest import (
    WeightsPackageError,
    find_weights_root,
    load_manifest,
    validate_manifest,
)


def _find_dinov2(base: Path = Path("/kaggle/input")) -> Path | None:
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "config.json" in files and "dinov2" in root.lower():
            return Path(root)
    return None


def _find_data_root(base: Path = Path("/kaggle/input")) -> Path:
    candidates = [
        Path("/kaggle/input/competitions") / config.COMPETITION_SLUG,
        Path("/kaggle/input") / config.COMPETITION_SLUG,
        config.DATA_DIR,
    ]
    for candidate in candidates:
        if (candidate / "test.csv").is_file() and (candidate / "test_series").is_dir():
            return candidate
    if base.is_dir():
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("train_series",)]
            path = Path(root)
            if "test.csv" in files and (path / "test_series").is_dir():
                return path
    raise FileNotFoundError(
        "competition mount não encontrado; esperado test.csv e test_series/"
    )


def _write_fallback(out: Path) -> pd.DataFrame:
    test_df = pd.read_csv(config.TEST_CSV)
    preds = np.full((len(test_df), len(config.TARGET_COLUMNS)), 0.5, dtype=np.float32)
    return write_submission(test_df[config.ID_COLUMN].tolist(), preds, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=None,
        help="Pasta com manifest.json do dataset pilkwang/rsna-knee-weights. "
             "Se omitido no Kaggle, procura automaticamente em /kaggle/input.",
    )
    parser.add_argument(
        "--dinov2-root",
        type=Path,
        default=None,
        help="Pasta do Kaggle Model metaresearch/dinov2/PyTorch/small/1.",
    )
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida manifest, inputs e escreve uma submissão 0.5; não carrega modelos/DICOM.",
    )
    args = parser.parse_args()

    weights_root = args.weights_root or find_weights_root()
    if weights_root is None:
        raise SystemExit(
            "Pacote de pesos não encontrado. Anexe o dataset público "
            "pilkwang/rsna-knee-weights ou passe --weights-root."
        )

    try:
        manifest = load_manifest(weights_root)
        validate_manifest(manifest)
    except WeightsPackageError as exc:
        raise SystemExit(f"Pacote de pesos incompatível: {exc}") from exc

    dinov2_root = args.dinov2_root or _find_dinov2()
    if dinov2_root is None:
        raise SystemExit(
            "DINOv2 não encontrado. Anexe o Kaggle Model "
            "metaresearch/dinov2/PyTorch/small/1 ou passe --dinov2-root."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"weights: {weights_root}")
    print(f"members: {len(manifest.members)} em {len(manifest.groups)} grupo(s) de pixel")
    print(f"dinov2: {dinov2_root}")
    print(f"device: {device}")
    if device.type != "cuda":
        print("AVISO: para reproduzir ~0.893, rode no Kaggle com GPU T4; CPU tende a cortar TTA.")

    data_root = _find_data_root()

    if args.dry_run:
        sub = _write_fallback(args.out)
        print(f"dry-run OK; submissão benchmark salva em {args.out} com shape {sub.shape}")
        return

    infer_from_manifest(
        manifest=manifest,
        data_root=data_root,
        dinov2_root=dinov2_root,
        out_path=args.out,
        device=device,
    )


if __name__ == "__main__":
    main()
