"""I/O de baixo nível para slices DICOM: leitura de arquivo, extração do
array de pixels, normalização de intensidade e redimensionamento.

Este módulo não decide lateralidade (ver `laterality.py`) nem qual
série/slice usar (ver `series.py`/`slices.py`) -- só sabe transformar um
caminho de arquivo `.dcm` num array de imagem pronto pra uso, sem nenhuma
lógica de negócio específica do desafio.
"""

from pathlib import Path

import cv2
import numpy as np
import pydicom


def read_dicom(path: Path) -> pydicom.Dataset:
    """Lê um arquivo DICOM do disco."""
    return pydicom.dcmread(str(path))


def read_pixel_array(ds: pydicom.Dataset) -> np.ndarray:
    """Extrai o array de pixels bruto (sem normalização) como float32."""
    return ds.pixel_array.astype(np.float32)


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Normalização min-max pra [0, 1]. O epsilon no denominador evita
    divisão por zero em imagens constantes (raro, mas não deve lançar
    exceção)."""
    rng = img.max() - img.min()
    return (img - img.min()) / (rng + 1e-6)


def resize_to(img: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona `img` pra (size, size) via interpolação padrão do cv2."""
    return cv2.resize(img, (size, size))


def read_pixel_spacing(ds: pydicom.Dataset) -> tuple[float, float] | None:
    """Lê `PixelSpacing` (mm/pixel, `[espaçamento entre linhas,
    espaçamento entre colunas]`) de `ds`. `None` (abstenção) se a tag não
    estiver presente ou for inválida -- nem todo DICOM garante essa tag,
    embora seja comum em MRI."""
    try:
        row_spacing, col_spacing = (float(v) for v in ds.PixelSpacing)
        return row_spacing, col_spacing
    except (AttributeError, TypeError, ValueError):
        return None


def crop_to_physical_fov(
    img: np.ndarray, pixel_spacing: tuple[float, float], fov_mm: float
) -> np.ndarray:
    """Recorta `img` centrado num campo de visão de `fov_mm` milímetros
    REAIS, calibrado por `pixel_spacing` -- não por uma fração fixa de
    pixels. Sem isso, a mesma estrutura anatômica ocupa uma fração
    diferente da imagem final dependendo do FOV de aquisição do scanner
    (que varia bastante entre estudos); duas imagens de scanners com
    `PixelSpacing` diferentes passam a representar a mesma extensão
    anatômica real em mm, não a mesma proporção da imagem original.

    Se `fov_mm` for maior que a imagem original em alguma dimensão
    (FOV de aquisição menor que o alvo), aplica padding replicando a
    borda antes de recortar -- evita lançar exceção nesse caso raro, à
    custa de inventar uma pequena faixa de borda repetida."""
    row_spacing, col_spacing = pixel_spacing
    crop_h = max(1, round(fov_mm / row_spacing))
    crop_w = max(1, round(fov_mm / col_spacing))
    h, w = img.shape

    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)
    if pad_h or pad_w:
        img = np.pad(
            img,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="edge",
        )
        h, w = img.shape

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return img[top:top + crop_h, left:left + crop_w]
