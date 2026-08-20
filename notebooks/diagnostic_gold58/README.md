# Diagnóstico: modelo de imagem vs. extrator de texto, nos 58 gold

**Não é a linha de submissão real** — kernel privado, só pra diagnóstico
local (não gasta submissão). Não confundir com `notebooks/kaggle_ativo/`.

## O que é

Cópia gerada de `notebooks/kaggle_ativo/rsna-knee-dino-radimagenet-rank-ensemble.ipynb`
(via `scripts/build_gold58_diagnostic_notebook.py`, que não edita o
notebook original) com duas mudanças:

1. Célula 0 nova: constrói um "mirror" da pasta da competição em
   `/kaggle/working/gold58_mirror/` — tudo symlinked do dataset real da
   competição, exceto `test.csv`/`test_series/`/`sample_submission.csv`,
   substituídos pelos 58 estudos "gold" (12 colunas de label preenchidas
   em `train.csv`) e suas séries reais (`train_series/`).
2. `ROOT`/`COMP` (linhas que apontavam pra pasta real da competição, nas
   células 1 e 2 originais) trocados pra apontar pro mirror.
3. Última célula nova: lê o `submission.csv` final gerado pelas 3 células
   de inferência (que rodam sem saber que estão vendo o mirror) e calcula
   AUC macro contra o gabarito real dos 58 gold.

Todo o resto (lógica de inferência, pesos, checkpoints, arms) é idêntico
ao kernel de submissão real — só a fonte de dados muda.

## Como regenerar

```
python scripts/build_gold58_diagnostic_notebook.py
```

Regenera `rsna-knee-gold58-diagnostic.ipynb` a partir do estado atual de
`notebooks/kaggle_ativo/`. Rodar de novo sempre que a linha ativa mudar,
se quiser repetir este diagnóstico.

## Como rodar

```
kaggle kernels push -p notebooks/diagnostic_gold58
```

Depois abrir no Kaggle, **conferir GPU T4** (mesma regra da linha ativa),
"Save & Run All". Resultado: duas listas de AUC por target impressas na
última célula (12 linhas + AUC macro).

## Resultado do lado do texto (referência, calculado local)

`python scripts/diagnose_text_vs_image_ceiling.py` → **AUC macro = 0.8073**
(extrator de regras sobre `Report`, nos mesmos 58 gold). Ver `PROGRESS.md`
pra comparação final com o AUC do modelo de imagem.
