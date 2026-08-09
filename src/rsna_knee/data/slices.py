"""Seleção de slice dentro de uma série (cada série tem 20-45 slices
DICOM).

Roadmap (ver CLAUDE.md/PROGRESS.md): hoje pega o slice do meio, ordenando
os arquivos por NOME -- não há garantia de que a ordem de arquivo
corresponda à posição anatômica real. Ordenação de slice pela geometria
DICOM (eixo dominante do paciente) é uma técnica futura que substitui a
ordenação usada aqui, sem mudar a interface deste módulo.
"""

from pathlib import Path


def pick_middle_slice(series_dir: Path) -> Path | None:
    """Retorna o arquivo `.dcm` do meio da série, em ordem alfabética de
    nome de arquivo. `None` se a pasta não tiver nenhum `.dcm`."""
    slices = sorted(series_dir.glob("*.dcm"))
    if not slices:
        return None
    return slices[len(slices) // 2]
