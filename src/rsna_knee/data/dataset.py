"""
`Dataset` do PyTorch (imagem) para o desafio RSNA Knee -- composição de
alto nível sobre `series.py` (seleção de série), `slices.py` (seleção de
slice) e `laterality.py` (carregamento + normalização de lateralidade).

Modelo final é image-only (ver `modeling/`) -- o texto do laudo (Report) só
é usado como weak supervision para gerar pseudo-labels (`labels.py`), nunca
como input do modelo: ele não existe em test.csv, então dependeria de um
sinal ausente no teste. Este Dataset não carrega/tokeniza o Report.

Cada STUDY (StudyInstanceUID) tem várias SÉRIES (SeriesInstanceUID), e cada
série tem 20-45 slices DICOM. Isso é bem mais complexo que "uma imagem por
estudo" -- a v1 abaixo faz uma simplificação deliberada para ter um baseline
rodando rápido:

    1 estudo -> escolhe 1 série (a sagital fluid-sensitive, via
    train_series.csv/test_series.csv -- ver series.select_preferred_series_id;
    cai de volta para a primeira série em ordem alfabética se não achar)
    -> pega o slice do meio dessa série -> 1 imagem 2D -> normaliza o lado
    do joelho (ver laterality.py) pros 4 labels medial/lateral fazerem
    sentido de forma consistente entre estudos de joelho esquerdo e
    direito.

Isso definitivamente deixa sinal na mesa (ignora as outras séries e todos os
outros slices). Ver o roadmap de técnicas futuras em CLAUDE.md/PROGRESS.md:
ordenação de slice por geometria, corte por escala física, seleção
multi-série com atenção, TTA/ensemble.

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

import numpy as np
import pandas as pd

from .. import config
from . import series as series_mod
from . import slices as slices_mod
from .laterality import load_dicom_slice


def load_study_image(
    study_dir: Path,
    study_id: str | None = None,
    series_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Monta a imagem (3, H, W) de um estudo: escolhe série (`series.py`),
    escolhe slice (`slices.py`), carrega + normaliza lateralidade
    (`laterality.py`), e replica em 3 canais pra bater com backbones
    pré-treinados em imagens RGB."""
    series_dir = series_mod.pick_series_dir(study_dir, study_id, series_df)
    if series_dir is None:
        raise FileNotFoundError(f"Nenhuma série encontrada em {study_dir}")
    slice_path = slices_mod.pick_middle_slice(series_dir)
    if slice_path is None:
        raise FileNotFoundError(f"Nenhum slice .dcm encontrado em {series_dir}")
    img = load_dicom_slice(slice_path)
    return np.stack([img] * 3, axis=0)  # (3, H, W) para backbones pré-treinados


def attach_label_weights(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Prepara `df` pra entrar no KneeDataset com peso por elemento (estudo x
    label) no loss -- usado pra dar menos peso às pseudo-labels de weak
    supervision (ver `labels.py`) do que aos 58 labels reais. Cria uma
    coluna "<label>__w" por target: `weight` onde o label está presente,
    0.0 onde está NaN (pseudo-label que a regra absteve, ou label excluído
    de EXCLUDED_FROM_PSEUDO_LABELS). Os NaN dos alvos em si são preenchidos
    com 0.0 (valor arbitrário -- peso 0 os anula no loss) só pra não
    propagar NaN pro tensor.

    Roadmap (CLAUDE.md): ponderação granular por confiança (peso por
    label, não um escalar único pra tudo) é uma técnica futura que estende
    esta função -- ver `labels.evaluate_against_gold`."""
    out = df.copy()
    for col in config.TARGET_COLUMNS:
        out[f"{col}{config.WEIGHT_COLUMN_SUFFIX}"] = out[col].notna().astype(np.float32) * weight
        out[col] = out[col].fillna(0.0)
    return out


class KneeDataset(Dataset):
    """
    Espera um DataFrame com pelo menos:
      - config.ID_COLUMN (StudyInstanceUID)
      - config.TARGET_COLUMNS -- presentes só se `is_train=True`
    """

    def __init__(
        self,
        df: pd.DataFrame,
        series_dir: Path,
        is_train: bool = True,
        series_csv: Path | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.series_dir = Path(series_dir)
        self.is_train = is_train
        # train_series.csv/test_series.csv (Anatomical_Plane, Fluid_Sensitive)
        # para escolher a série sagital fluid-sensitive em vez da primeira
        # série disponível. Sem isso, cai de volta pra ordem alfabética.
        self.series_df = pd.read_csv(series_csv) if series_csv is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        study_id = row[config.ID_COLUMN]

        study_dir = self.series_dir / str(study_id)
        image = torch.from_numpy(load_study_image(study_dir, study_id, self.series_df))

        item = {
            "study_id": study_id,
            "image": image,
        }

        if self.is_train:
            targets = row[config.TARGET_COLUMNS].astype(np.float64).fillna(0.0).values.astype(np.float32)
            item["targets"] = torch.from_numpy(targets)

            weight_cols = [f"{c}{config.WEIGHT_COLUMN_SUFFIX}" for c in config.TARGET_COLUMNS]
            if all(c in self.df.columns for c in weight_cols):
                weights = row[weight_cols].values.astype(np.float32)
            else:
                # sem attach_label_weights (ex.: uso direto fora do fluxo de
                # weak supervision) -- todo mundo pesa igual, como antes.
                weights = np.ones(len(config.TARGET_COLUMNS), dtype=np.float32)
            item["target_weights"] = torch.from_numpy(weights)

        return item
