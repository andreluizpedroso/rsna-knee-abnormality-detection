"""Seleção de slice dentro de uma série (cada série tem 20-45 slices
DICOM).

Ordena os slices pela posição real ao longo do eixo de empilhamento da
série (geometria DICOM via `ImagePositionPatient`/`ImageOrientationPatient`),
não pela ordem alfabética de nome de arquivo -- essa não tem garantia de
corresponder à posição anatômica real (SOPInstanceUID não é necessariamente
monotônico com a posição). Roadmap (ver CLAUDE.md): próxima extensão natural
é amostrar múltiplos slices ao redor do centro (2.5D/TTA), não só o do meio;
a ordenação geométrica é pré-requisito pra isso -- empilhar slices fora de
ordem anatômica misturaria estruturas de posições diferentes.
"""

from pathlib import Path

import numpy as np
import pydicom


def _slice_position(path: Path) -> float | None:
    """Projeta `ImagePositionPatient` no vetor normal ao plano de corte
    (produto vetorial dos dois vetores de `ImageOrientationPatient`) -- a
    projeção escalar dá a posição do slice ao longo do eixo de
    empilhamento da série. Mais robusto que `InstanceNumber` (que pode
    estar ausente ou não-monotônico) porque lê a geometria real do slice,
    não um contador atribuído pelo scanner/exportador.

    Só lê o header (`stop_before_pixels=True`) -- não precisa decodificar
    o pixel_array pra ordenar, então é barato mesmo para séries grandes.
    Retorna `None` (abstenção) se a geometria não estiver disponível ou
    for inválida nesse arquivo."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        ipp = np.array([float(v) for v in ds.ImagePositionPatient])
        iop = [float(v) for v in ds.ImageOrientationPatient]
        row_vec = np.array(iop[0:3])
        col_vec = np.array(iop[3:6])
        normal = np.cross(row_vec, col_vec)
        return float(np.dot(ipp, normal))
    except (pydicom.errors.InvalidDicomError, AttributeError, TypeError, ValueError, IndexError, OSError):
        return None


def order_slices_by_geometry(series_dir: Path) -> list[Path]:
    """Ordena os `.dcm` de `series_dir` pela posição real ao longo do eixo
    de empilhamento da série (ver `_slice_position`).

    Se a geometria não estiver disponível pra QUALQUER slice da série, cai
    de volta pra ordem alfabética de nome de arquivo pra série INTEIRA --
    misturar slices ordenados por geometria com slices mantidos na ordem
    de arquivo produziria uma sequência sem sentido anatômico; abster e
    usar o fallback antigo (ordem de arquivo) é mais seguro que uma ordem
    parcialmente confiável e silenciosamente errada."""
    paths = sorted(series_dir.glob("*.dcm"))
    if not paths:
        return []

    positions = [_slice_position(p) for p in paths]
    if any(pos is None for pos in positions):
        return paths

    order = sorted(range(len(paths)), key=lambda i: positions[i])
    return [paths[i] for i in order]


def pick_middle_slice(series_dir: Path) -> Path | None:
    """Retorna o arquivo `.dcm` do meio da série, ordenado pela posição
    geométrica real (ver `order_slices_by_geometry`). `None` se a pasta
    não tiver nenhum `.dcm`."""
    slices = order_slices_by_geometry(series_dir)
    if not slices:
        return None
    return slices[len(slices) // 2]
