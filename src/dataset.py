"""
Dataset (imagem) para o desafio RSNA Knee.

Modelo final é image-only (ver model.py) -- o texto do laudo (Report) só é
usado como weak supervision para gerar pseudo-labels (weak_supervision.py),
nunca como input do modelo: ele não existe em test.csv, então dependeria de
um sinal ausente no teste. Este Dataset não carrega/tokeniza o Report.

Cada STUDY (StudyInstanceUID) tem várias SÉRIES (SeriesInstanceUID), e cada
série tem 20-45 slices DICOM. Isso é bem mais complexo que "uma imagem por
estudo" -- a v1 abaixo faz uma simplificação deliberada para ter um baseline
rodando rápido:

    1 estudo -> escolhe 1 série (a sagital fluid-sensitive, via
    train_series.csv/test_series.csv -- ver select_preferred_series_id;
    cai de volta para a primeira série em ordem alfabética se não achar)
    -> pega o slice do meio dessa série -> 1 imagem 2D -> normaliza o lado
    do joelho (ver compute_laterality/CANONICAL_LATERALITY abaixo, pros 4
    labels medial/lateral fazerem sentido de forma consistente entre
    estudos de joelho esquerdo e direito).

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


# Lado escolhido como referência -- séries do lado oposto são espelhadas
# horizontalmente (ver load_dicom_slice) pra normalizar o eixo
# medial-lateral. 4 dos 12 labels (Medial/Lateral Meniscus, Medial/Lateral
# OA) só fazem sentido em relação a um lado fixo do joelho; sem essa
# normalização, joelho esquerdo e direito têm o eixo medial-lateral
# espelhado um em relação ao outro, e o modelo aprenderia esses 4 labels a
# partir de um eixo invertido em ~metade dos estudos.
CANONICAL_LATERALITY = "R"


def compute_laterality(ds) -> str | None:
    """Determina o lado do joelho ('L' ou 'R') a partir da GEOMETRIA DICOM
    (ImagePositionPatient + ImageOrientationPatient), não da tag
    `Laterality` (0020,0060) diretamente -- essa tag é opcional no padrão
    DICOM e fica ausente/vazia em boa parte dos estudos (2 dos 5 estudos
    de smoke test locais, por exemplo); tratar ausência como um lado fixo
    por padrão espelharia silenciosamente metade dos estudos do lado
    oposto. A convenção de coordenadas do paciente do DICOM (LPS) tem X
    positivo apontando pro lado ESQUERDO do paciente -- então o sinal do X
    do centro da imagem (calculado a partir da geometria, não só do canto
    IPP) indica o lado. Validado contra a tag Laterality nos estudos locais
    onde ambas estão disponíveis (bateu 100% -- ver PROGRESS.md).

    Cai de volta pra tag `Laterality` só quando a geometria não está
    disponível (raro -- IPP/IOP são campos praticamente sempre presentes
    em MRI). Retorna None (abstenção -- não espelha) se nenhuma das duas
    fontes permitir decidir."""
    try:
        ipp = np.array([float(v) for v in ds.ImagePositionPatient])
        iop = [float(v) for v in ds.ImageOrientationPatient]
        rows, cols = int(ds.Rows), int(ds.Columns)
        px_spacing = [float(v) for v in ds.PixelSpacing]  # [row_spacing, col_spacing]
        row_vec = np.array(iop[0:3])  # direção ao longo de uma LINHA (índice de coluna crescente)
        col_vec = np.array(iop[3:6])  # direção ao longo de uma COLUNA (índice de linha crescente)
        center_x = (
            ipp + (cols / 2) * px_spacing[1] * row_vec + (rows / 2) * px_spacing[0] * col_vec
        )[0]
        if center_x > 0:
            return "L"
        if center_x < 0:
            return "R"
        # center_x == 0 (raríssimo, exatamente na linha média) -- ambíguo,
        # cai pro fallback abaixo.
    except (AttributeError, TypeError, ValueError, IndexError):
        pass

    laterality = getattr(ds, "Laterality", None)
    return laterality if laterality in ("L", "R") else None


def load_dicom_slice(path: Path) -> np.ndarray:
    """Carrega um único slice DICOM como array float32 normalizado em [0,1].
    Espelha horizontalmente se o lado do joelho (ver compute_laterality) não
    for o CANONICAL_LATERALITY -- ver comentário acima da constante.

    Nota: essa normalização corrige diretamente o eixo medial-lateral em
    séries CORONAIS/AXIAIS, onde esquerda-direita na imagem 2D corresponde
    a medial-lateral anatômico. Em séries SAGITAIS (as priorizadas por
    select_preferred_series_id hoje), esquerda-direita na imagem 2D
    corresponde a anterior-posterior, não a medial-lateral -- pra essas, o
    espelhamento troca o enquadramento anterior/posterior em vez de
    corrigir medial/lateral. Ainda assim aplicamos de forma consistente
    (não condicionamos ao plano) porque: (1) é inofensivo pra sagital
    (segue sendo anatomia válida, só com A/P trocado) e (2) já deixa o
    pipeline correto pra quando series coronais/axiais entrarem em uso
    (fallback atual, ou uma evolução futura multi-plano)."""
    import pydicom
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    rng = img.max() - img.min()
    img = (img - img.min()) / (rng + 1e-6)

    side = compute_laterality(ds)
    if side is not None and side != CANONICAL_LATERALITY:
        img = np.ascontiguousarray(img[:, ::-1])

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
      - config.TARGET_COLUMNS -- presentes só se `is_train=True`
    """

    def __init__(
        self,
        df: pd.DataFrame,
        series_dir: Path,
        is_train: bool = True,
        series_csv: Path | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.series_dir = Path(series_dir)
        self.is_train = is_train
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

        item = {
            "study_id": study_id,
            "image": image,
        }

        if self.is_train:
            targets = row[config.TARGET_COLUMNS].astype(np.float64).fillna(0.0).values.astype(np.float32)
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
