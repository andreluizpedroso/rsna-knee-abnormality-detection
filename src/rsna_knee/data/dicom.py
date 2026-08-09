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
