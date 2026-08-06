"""
Dataset multimodal (imagem + texto do laudo) para o desafio RSNA Knee.

Cada STUDY (StudyInstanceUID) tem várias SÉRIES (SeriesInstanceUID), e cada
série tem 20-45 slices DICOM. Isso é bem mais complexo que "uma imagem por
estudo" -- a v1 abaixo faz uma simplificação deliberada para ter um baseline
rodando rápido:

    1 estudo -> escolhe 1 série (a sagital fluid-sensitive, via
    train_series.csv/test_series.csv -- ver select_preferred_series_id;
    cai de volta para a primeira série em ordem alfabética se não achar)
    -> pega o slice do meio dessa série -> 1 imagem 2D.

Isso definitivamente deixa sinal na mesa (ignora as outras séries e todos os
outros slices). Depois que o baseline estiver validado, vale evoluir para:
  - 2.5D: empilhar N slices centrais como canais
  - Multi-série: um encoder de imagem por plano anatômico, fundidos
  - MIL (multiple instance learning) sobre todos os slices/séries do estudo

IMPORTANTE (rótulos): só uma parte dos estudos de treino tem as 12 colunas de
label preenchidas. Este Dataset assume que `df` já veio filtrado/preparado
(ex.: `df.dropna(subset=config.TARGET_COLUMNS)` para treino supervisionado
direto, ou nenhum filtro se for usar os Reports para weak supervision).
"""

from pathlib import Path

# No Windows, importar torch depois de pandas pode quebrar o carregamento de
# DLL do torch (c10.dll) por conflito de dependências nativas -- torch
# precisa vir primeiro.
import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import pandas as pd

from . import config


def load_dicom_slice(path: Path) -> np.ndarray:
    """Carrega um único slice DICOM como array float32 normalizado em [0,1]."""
    import pydicom
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    rng = img.max() - img.min()
    img = (img - img.min()) / (rng + 1e-6)
    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return img


def select_preferred_series_id(study_id: str, series_df: pd.DataFrame) -> str | None:
    """Escolhe a SeriesInstanceUID mais informativa para um estudo, usando
    train_series.csv/test_series.csv. Prioridade: sagital + fluid-sensitive
    (melhor visualização de ligamento/menisco) > sagital > fluid-sensitive
    em qualquer plano > qualquer série disponível."""
    candidates = series_df[series_df[config.ID_COLUMN] == study_id]
    if candidates.empty:
        return None

    is_sagittal = candidates["Anatomical_Plane"] == "Sagittal"
    is_fluid_sensitive = candidates["Fluid_Sensitive"] == 1
    score = is_sagittal.astype(int) * 2 + is_fluid_sensitive.astype(int)

    best_idx = score.idxmax()
    return candidates.loc[best_idx, config.SERIES_ID_COLUMN]


def pick_series_dir(
    study_dir: Path,
    study_id: str | None = None,
    series_df: pd.DataFrame | None = None,
) -> Path | None:
    """Escolhe uma série (pasta) dentro de um estudo. Se `series_df` (de
    train_series.csv/test_series.csv) e `study_id` forem passados, prioriza a
    série sagital fluid-sensitive (ver select_preferred_series_id). Caso
    contrário, ou se a série preferida não estiver na pasta baixada, cai de
    volta para a primeira série em ordem alfabética."""
    if study_id is not None and series_df is not None:
        preferred_id = select_preferred_series_id(study_id, series_df)
        if preferred_id is not None:
            preferred_dir = study_dir / str(preferred_id)
            if preferred_dir.is_dir():
                return preferred_dir

    series_dirs = sorted(p for p in study_dir.iterdir() if p.is_dir())
    return series_dirs[0] if series_dirs else None


def pick_middle_slice(series_dir: Path) -> Path | None:
    slices = sorted(series_dir.glob("*.dcm"))
    if not slices:
        return None
    return slices[len(slices) // 2]


def load_study_image(
    study_dir: Path,
    study_id: str | None = None,
    series_df: pd.DataFrame | None = None,
) -> np.ndarray:
    series_dir = pick_series_dir(study_dir, study_id, series_df)
    if series_dir is None:
        raise FileNotFoundError(f"Nenhuma série encontrada em {study_dir}")
    slice_path = pick_middle_slice(series_dir)
    if slice_path is None:
        raise FileNotFoundError(f"Nenhum slice .dcm encontrado em {series_dir}")
    img = load_dicom_slice(slice_path)
    return np.stack([img] * 3, axis=0)  # (3, H, W) para backbones pré-treinados


WEIGHT_COLUMN_SUFFIX = "__w"


def attach_label_weights(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Prepara `df` pra entrar no KneeDataset com peso por elemento (estudo x
    label) no loss -- usado pra dar menos peso às pseudo-labels de weak
    supervision (ver src/weak_supervision.py) do que aos 58 labels reais.
    Cria uma coluna "<label>__w" por target: `weight` onde o label está
    presente, 0.0 onde está NaN (pseudo-label que a regra absteve, ou label
    excluído de EXCLUDED_FROM_PSEUDO_LABELS). Os NaN dos alvos em si são
    preenchidos com 0.0 (valor arbitrário -- peso 0 os anula no loss) só pra
    não propagar NaN pro tensor."""
    out = df.copy()
    for col in config.TARGET_COLUMNS:
        out[f"{col}{WEIGHT_COLUMN_SUFFIX}"] = out[col].notna().astype(np.float32) * weight
        out[col] = out[col].fillna(0.0)
    return out


class KneeDataset(Dataset):
    """
    Espera um DataFrame com pelo menos:
      - config.ID_COLUMN (StudyInstanceUID)
      - config.TEXT_COLUMN (Report) -- pode ser vazio/NaN
      - config.TARGET_COLUMNS -- presentes só se `is_train=True`
    """

    def __init__(
        self,
        df: pd.DataFrame,
        series_dir: Path,
        tokenizer=None,
        is_train: bool = True,
        max_text_len: int = 256,
        series_csv: Path | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.series_dir = Path(series_dir)
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_text_len = max_text_len
        # train_series.csv/test_series.csv (Anatomical_Plane, Fluid_Sensitive)
        # para escolher a série sagital fluid-sensitive em vez da primeira
        # série disponível. Sem isso, cai de volta pra ordem alfabética.
        self.series_df = pd.read_csv(series_csv) if series_csv is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        study_id = row[config.ID_COLUMN]

        study_dir = self.series_dir / str(study_id)
        image = torch.from_numpy(load_study_image(study_dir, study_id, self.series_df))

        text = str(row.get(config.TEXT_COLUMN, "") or "")
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        item = {
            "study_id": study_id,
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.is_train:
            targets = row[config.TARGET_COLUMNS].fillna(0.0).values.astype(np.float32)
            item["targets"] = torch.from_numpy(targets)

            weight_cols = [f"{c}{WEIGHT_COLUMN_SUFFIX}" for c in config.TARGET_COLUMNS]
            if all(c in self.df.columns for c in weight_cols):
                weights = row[weight_cols].values.astype(np.float32)
            else:
                # sem attach_label_weights (ex.: uso direto fora do fluxo de
                # weak supervision) -- todo mundo pesa igual, como antes.
                weights = np.ones(len(config.TARGET_COLUMNS), dtype=np.float32)
            item["target_weights"] = torch.from_numpy(weights)

        return item
