"""Gera notebooks/diagnostic_gold58/*.ipynb a partir de
notebooks/kaggle_ativo/rsna-knee-dino-radimagenet-rank-ensemble.ipynb,
sem editar o notebook original.

O que muda:
1. Insere uma celula de setup no inicio que constroi um "mirror" da pasta
   da competicao em /kaggle/working/gold58_mirror -- tudo symlinked do
   real, exceto test.csv/test_series/sample_submission.csv, que sao
   substituidos pelo subconjunto dos 58 estudos "gold" (12 labels
   preenchidas em train.csv), puxados de train.csv/train_series.
2. Troca a linha `ROOT = Path('/kaggle/input/competitions/...')` (celula
   1) por `ROOT = MIRROR`.
3. Troca a linha `COMP = Path('/kaggle/input/competitions/...')` (celula
   2) por `COMP = MIRROR`.

Roda 100% via manipulacao do JSON do .ipynb -- nao precisa reabrir o
notebook original no Jupyter.
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("notebooks/kaggle_ativo/rsna-knee-dino-radimagenet-rank-ensemble.ipynb")
DST_DIR = Path("notebooks/diagnostic_gold58")
DST = DST_DIR / "rsna-knee-gold58-diagnostic.ipynb"

SETUP_SOURCE = '''from pathlib import Path
import os
import pandas as pd

REAL = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
MIRROR = Path('/kaggle/working/gold58_mirror')
MIRROR.mkdir(parents=True, exist_ok=True)

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']

train_df = pd.read_csv(REAL / 'train.csv')
gold = train_df.set_index('StudyInstanceUID')[TARGETS]
gold = gold[gold.notna().all(axis=1)]
gold_uids = gold.index.tolist()
print(f'Estudos gold (12 labels preenchidas): {len(gold_uids)}')

for item in REAL.iterdir():
    if item.name in ('test.csv', 'test_series', 'test_series.csv', 'sample_submission.csv'):
        continue
    dst = MIRROR / item.name
    if not dst.exists():
        os.symlink(item, dst)

pd.DataFrame({'StudyInstanceUID': gold_uids}).to_csv(MIRROR / 'test.csv', index=False)

sub = pd.DataFrame({'StudyInstanceUID': gold_uids})
for t in TARGETS:
    sub[t] = 0.5
sub.to_csv(MIRROR / 'sample_submission.csv', index=False)

train_series = pd.read_csv(REAL / 'train_series.csv')
gold_series = train_series[train_series['StudyInstanceUID'].isin(gold_uids)]
gold_series.to_csv(MIRROR / 'test_series.csv', index=False)

(MIRROR / 'test_series').mkdir(exist_ok=True)
for uid in gold_uids:
    src = REAL / 'train_series' / uid
    dst = MIRROR / 'test_series' / uid
    if src.exists() and not dst.exists():
        os.symlink(src, dst)

print(f'Mirror pronto em {MIRROR} (test.csv/test_series/sample_submission.csv '
      f'substituidos pelos {len(gold_uids)} estudos gold; resto symlinked do real)')
'''

SCORE_SOURCE = '''import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']

pred = pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
gold_full = pd.read_csv(REAL / 'train.csv')
gold_full = gold_full.set_index('StudyInstanceUID')[TARGETS]
gold_full = gold_full.loc[gold_full.notna().all(axis=1)]

pred = pred.set_index('StudyInstanceUID').loc[gold_full.index, TARGETS]

per_target_auc = {}
for tgt in TARGETS:
    y_true = gold_full[tgt].values
    y_score = pred[tgt].values
    if len(np.unique(y_true)) < 2:
        print(f'  [aviso] {tgt}: so uma classe no gold (58), pulando AUC')
        continue
    per_target_auc[tgt] = roc_auc_score(y_true, y_score)

print('AUC por target (modelo de imagem vs. gold):')
for tgt, auc in per_target_auc.items():
    print(f'  {tgt:20s} {auc:.4f}')

macro_auc = float(np.mean(list(per_target_auc.values())))
print(f'\\nAUC MACRO (modelo de imagem, {len(per_target_auc)}/{len(TARGETS)} targets): {macro_auc:.4f}')
'''


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    code_cell_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    assert len(code_cell_idx) == 3, f"esperava 3 celulas de codigo, achei {len(code_cell_idx)}"
    c1, c2, c3 = code_cell_idx

    def cell_src(i):
        return "".join(nb["cells"][i]["source"])

    def set_cell_src(i, new_src):
        nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None

    src1 = cell_src(c1)
    needle1 = "ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')"
    assert needle1 in src1, "linha ROOT= nao encontrada na celula 1 (fonte pode ter mudado)"
    src1 = src1.replace(needle1, "ROOT = MIRROR  # [diagnostico gold58] era: " + needle1)
    set_cell_src(c1, src1)

    src2 = cell_src(c2)
    needle2 = "COMP = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')"
    assert needle2 in src2, "linha COMP= nao encontrada na celula 2 (fonte pode ter mudado)"
    src2 = src2.replace(needle2, "COMP = MIRROR  # [diagnostico gold58] era: " + needle2)
    set_cell_src(c2, src2)

    setup_cell = make_code_cell(SETUP_SOURCE)
    nb["cells"].insert(0, setup_cell)
    nb["cells"].append(make_code_cell(SCORE_SOURCE))

    DST_DIR.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook diagnostico escrito em {DST}")


if __name__ == "__main__":
    main()
