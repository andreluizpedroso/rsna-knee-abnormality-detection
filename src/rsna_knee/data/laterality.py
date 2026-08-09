"""Normalização de lateralidade: determina o lado do joelho (esquerdo/
direito) a partir da geometria DICOM e orquestra o carregamento de um slice
já normalizado nesse eixo, reaproveitando a leitura/redimensionamento de
baixo nível de `dicom.py`.

Roadmap (ver CLAUDE.md/PROGRESS.md): esta técnica já está implementada;
trabalho futuro é robustecê-la (validação sistemática de divergência
geometria x tag em escala, e possivelmente condicionar o espelhamento ao
plano anatômico da série) -- ver `CLAUDE.md`, item de roadmap de
lateralidade.
"""

from pathlib import Path

import numpy as np
import pydicom

from .. import config
from . import dicom as dicom_io


def compute_laterality(ds: pydicom.Dataset) -> str | None:
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


def apply_laterality_normalization(img: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Espelha `img` horizontalmente se o lado do joelho computado a partir
    de `ds` (ver `compute_laterality`) não for `config.CANONICAL_LATERALITY`.

    Nota: essa normalização corrige diretamente o eixo medial-lateral em
    séries CORONAIS/AXIAIS, onde esquerda-direita na imagem 2D corresponde
    a medial-lateral anatômico. Em séries SAGITAIS (as priorizadas por
    `series.select_preferred_series_id` hoje), esquerda-direita na imagem
    2D corresponde a anterior-posterior, não a medial-lateral -- pra essas,
    o espelhamento troca o enquadramento anterior/posterior em vez de
    corrigir medial/lateral. Ainda assim aplicamos de forma consistente
    (não condicionamos ao plano) porque: (1) é inofensivo pra sagital
    (segue sendo anatomia válida, só com A/P trocado) e (2) já deixa o
    pipeline correto pra quando séries coronais/axiais entrarem em uso
    (fallback atual, ou uma evolução futura multi-plano)."""
    side = compute_laterality(ds)
    if side is not None and side != config.CANONICAL_LATERALITY:
        return np.ascontiguousarray(img[:, ::-1])
    return img


def load_dicom_slice(path: Path, image_size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Carrega um único slice DICOM como array float32 normalizado em
    [0,1], com lateralidade corrigida (ver `apply_laterality_normalization`)
    e redimensionado pra `image_size`. Pipeline completo usado por
    `dataset.load_study_image`."""
    ds = dicom_io.read_dicom(path)
    img = dicom_io.normalize_intensity(dicom_io.read_pixel_array(ds))
    img = apply_laterality_normalization(img, ds)
    return dicom_io.resize_to(img, image_size)
