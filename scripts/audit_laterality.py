"""
Auditoria de divergência entre a lateralidade computada por geometria DICOM
(`compute_laterality`) e a tag `Laterality` (0020,0060), no dataset local
disponível. Ferramenta pontual (não chamada em treino/inferência) -- ver
CLAUDE.md, roadmap item 2 ("robustecer lateralidade").

Uso:
    python scripts/audit_laterality.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Mesma ordem de import do resto do projeto -- no Windows, importar torch
# depois de pandas pode quebrar o carregamento de DLL do torch (c10.dll).
import torch  # noqa: F401

import pandas as pd
import pydicom

from src.rsna_knee import config
from src.rsna_knee.data.laterality import compute_laterality
from src.rsna_knee.data.series import lookup_anatomical_plane


def audit(series_root: Path, series_df: pd.DataFrame) -> pd.DataFrame:
    """Roda `compute_laterality` num slice de cada série disponível
    localmente e compara com a tag `Laterality`. Retorna 1 linha por
    série auditada."""
    rows = []
    for study_dir in sorted(p for p in series_root.iterdir() if p.is_dir()):
        for series_dir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
            dcm_files = sorted(series_dir.glob("*.dcm"))
            if not dcm_files:
                continue
            ds = pydicom.dcmread(str(dcm_files[len(dcm_files) // 2]), stop_before_pixels=True)
            geometry_side = compute_laterality(ds)
            tag_side = getattr(ds, "Laterality", None)
            tag_side = tag_side if tag_side in ("L", "R") else None
            plane = lookup_anatomical_plane(series_dir.name, series_df)
            rows.append({
                "study_id": study_dir.name,
                "series_id": series_dir.name,
                "plane": plane,
                "geometry_side": geometry_side,
                "tag_side": tag_side,
                "concordant": (
                    geometry_side == tag_side
                    if geometry_side is not None and tag_side is not None
                    else None
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    series_df = pd.read_csv(config.TRAIN_SERIES_CSV)
    report = audit(config.TRAIN_SERIES_DIR, series_df)

    if report.empty:
        print("Nenhuma série encontrada localmente -- baixe uma amostra com "
              "scripts/download_sample_images.py antes de rodar a auditoria.")
        return

    print(report.to_string(index=False))

    both_available = report[report["concordant"].notna()]
    print(f"\n{len(report)} série(s) auditada(s).")
    print(f"{len(both_available)} com geometria E tag disponíveis.")
    if len(both_available):
        rate = both_available["concordant"].mean()
        n_ok = int(both_available["concordant"].sum())
        print(f"Concordância geometria x tag: {rate*100:.1f}% ({n_ok}/{len(both_available)})")

    n_no_tag = int(report["tag_side"].isna().sum())
    print(f"Série(s) sem tag Laterality (dependem só da geometria): {n_no_tag}/{len(report)}")


if __name__ == "__main__":
    main()
