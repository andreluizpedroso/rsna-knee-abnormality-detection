# RSNA Knee Abnormality Detection

Modelo de visão computacional para detectar 12 anormalidades clinicamente
relevantes no joelho a partir de MRI, desenvolvido para a competição Kaggle
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

## O desafio

- **Entrada**: séries de ressonância magnética (DICOM) do joelho, por
  estudo, com múltiplas séries/planos anatômicos por estudo.
- **Saída**: probabilidade (0–1) de 12 anormalidades por estudo — ACL, MCL,
  Meniscus (medial/lateral), OA (medial/lateral/patellofemoral), Effusion,
  Synovitis, Baker's cyst, Contusion, Fracture.
- **Métrica**: AUC-ROC macro entre os 12 targets.
- **Formato**: *code competition* — a submissão final roda dentro de um
  Kaggle Notebook, sem acesso à internet durante a execução, runtime ≤ 9h.
- **Prêmio**: existe uma trilha adicional de eficiência que pondera AUC vs.
  tempo de execução, o que motiva manter o pipeline enxuto.

## Pipeline ativo: `src/rsna_knee/`

O pipeline em desenvolvimento ativo é o pacote próprio `src/rsna_knee/`:
weak supervision por regras sobre o `Report`, seleção de série
sagital fluid-sensitive, normalização de lateralidade via geometria DICOM,
encoder DINOv2-small + cabeça de classificação para as 12 classes.

Um notebook público de terceiros (licença Apache 2.0) foi usado
temporariamente como pipeline de submissão pra aprender técnicas gerais de
processamento de imagem médica/DICOM que o pipeline próprio ainda não
tinha (ver "Histórico de submissões públicas" abaixo) — o objetivo sempre
foi trazer essas técnicas de volta pro `src/rsna_knee/` em termos próprios,
não depender do código dele continuamente. Roadmap de técnicas a portar
(ordenação de slice por geometria, TTA, ensemble ponderado por holdout,
etc.) documentado no `CLAUDE.md`.

### Estrutura do pacote

```
src/rsna_knee/
  config.py           # paths, hiperparâmetros, lista de targets (centralizado)
  data/
    dicom.py           # I/O de baixo nível (leitura, normalização, resize)
    laterality.py      # normalização de lateralidade via geometria DICOM
    series.py          # seleção de série (sagital fluid-sensitive)
    slices.py          # seleção de slice dentro da série
    labels.py           # weak supervision (pseudo-labels a partir do Report)
    dataset.py          # Dataset/DataLoader do PyTorch
  modeling/
    backbone.py          # encoder de imagem (DINOv2 via timm)
    heads.py              # cabeça de classificação
    model.py               # composição backbone + head
    ensemble.py             # ensemble entre checkpoints (roadmap, vazio hoje)
  training/
    validation.py           # AUC-ROC macro
    loop.py                  # loop de treino/validação por fold
  inference/
    submission.py             # carrega checkpoints, ensemble, gera submission.csv
    tta.py                     # TTA de slice (roadmap, vazio hoje)
  cli/
    train.py                    # entrypoint: python -m src.rsna_knee.cli.train
    infer.py                     # entrypoint: python -m src.rsna_knee.cli.infer
    evaluate_labels.py            # avalia as regras de weak supervision
tests/                            # pytest, sem GPU/dados reais (sintéticos/mocks)
```

## Histórico de submissões públicas

Entre as sessões em que o pipeline próprio ficou pausado, um notebook
público (licença Apache 2.0) foi adotado como pipeline de submissão —
permitido pelas regras da competição, código público pode ser reaproveitado
(não pode ser compartilhado *privadamente* entre times). Isso gerou 4
submissões oficiais reais, que continuam válidas no leaderboard:

| Abordagem | publicScore |
|---|---|
| Cópia direta do notebook original (treina do zero, GPU T4) | 0.824 |
| Split treino/infer (rank-mean r224+r336) | 0.833 |
| Ensemble de pesos publicados + patch TTA, CPU | 0.878 |
| Ensemble de pesos publicados + patch TTA, GPU T4 | **0.893** |

Os notebooks reais que rodaram estão em `notebooks/ativo/04_ensemble_infer/`
e `notebooks/ativo/04b_continue_finetune/` (cada pasta com o `.ipynb`
pulled do Kaggle + `kernel-metadata.json`) — mantidos como **registro
histórico** de que rodaram e atingiram esses scores, não como alvo de
desenvolvimento daqui pra frente. Ver `KAGGLE_WORKFLOW.md` pra reproduzir
o resultado 0.893 se algum dia for útil de novo, e `NOTICE.md` pra
atribuição legal ao código de terceiros.

## Dados

- `train.csv`: uma linha por estudo (`StudyInstanceUID`, `Report`, 12
  colunas de label). Só uma pequena fração dos estudos de treino tem as 12
  colunas de label preenchidas — o restante vem sem label direto.
  `PatientSex` não é mais uma coluna do CSV (removida pelo host — já vem no
  header DICOM).
- `train_series.csv`: uma linha por série (`SeriesInstanceUID`,
  `Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression`).
- `Report`: texto livre do laudo radiológico, multi-idioma, presente em
  100% dos estudos de treino. **Não existe na base de teste** (confirmado
  pelo host) — por isso é usado só para gerar/checar labels adicionais no
  treino (weak supervision), nunca como input do modelo.
- O gabarito (labels "gold") vem sempre de leitura de imagem por
  radiologistas MSK, nunca do texto do laudo (confirmado pelo host) — todo
  extrator de weak supervision baseado em texto tem um teto embutido por
  isso.
- Dataset completo: 819.640 arquivos DICOM, 569.76 GB.

## Estrutura do projeto

```
data/                          # dados da competição (não versionado)
notebooks/
  historico/                    # notebooks .ipynb soltos do início do projeto
    01_eda.ipynb                  # exploração inicial dos dados
    02_train_kaggle.ipynb         # treino no Kaggle (versão antiga do pipeline)
    02b_refresh_submission_assets.ipynb
    03_submit_kaggle.ipynb
  ativo/                         # histórico de submissões públicas (ver seção acima)
    04_ensemble_infer/            # .ipynb + kernel-metadata.json (datasets/models/GPU
    04b_continue_finetune/        # anexados), reproduzível sozinho no Kaggle
src/rsna_knee/                  # PIPELINE ATIVO -- ver "Estrutura do pacote" acima
scripts/
  download_data.sh              # download dos dados via Kaggle API
  download_sample_images.py     # baixa amostra mínima pra --smoke-test
checkpoints/                    # saída local do treino (src.rsna_knee.cli.train);
                                 # *.pth gitignored, nunca vai pro GitHub
submissions/                    # submissões geradas localmente
tests/                          # pytest
KAGGLE_WORKFLOW.md              # como reproduzir o histórico de submissões públicas
NOTICE.md                       # atribuição legal a código de terceiros reaproveitado
```

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Baixar os dados (requer aceitar as regras da competição no Kaggle e ter
`kaggle.json` configurado):

```bash
bash scripts/download_data.sh          # CSVs (metadados/labels/reports)
bash scripts/download_data.sh --full   # dataset completo (569.76 GB)
python scripts/download_sample_images.py --n 5   # amostra mínima pra --smoke-test
```

Treinar localmente numa amostra pequena (validação do pipeline):

```bash
python -m src.rsna_knee.cli.train --smoke-test --epochs 2
```

Treinar (fold completo):

```bash
python -m src.rsna_knee.cli.train
```

Gerar submissão a partir de um ou mais checkpoints (ensemble):

```bash
python -m src.rsna_knee.cli.infer --checkpoint checkpoints/best_fold0.pth
```

Avaliar as regras de weak supervision contra os estudos com label real:

```bash
python -m src.rsna_knee.cli.evaluate_labels
```

Rodar os testes:

```bash
pytest tests/
```

O treino em escala real acontece em um Kaggle Notebook (GPU gratuita, dados
já montados no ambiente) — é uma exigência da competição, já que a
submissão final não pode depender de acesso à internet nem de upload direto
de artefatos externos.

## Status

- Pipeline ativo (`src/rsna_knee/`): reestruturado em pacote modular,
  paridade de comportamento confirmada contra a versão anterior (mesmos
  arquivos soltos migrados 1:1). Melhor score oficial confirmado até agora
  com esta lógica: 0.604 (macro AUC-ROC, publicScore) — roadmap de técnicas
  novas em `CLAUDE.md`.
- Histórico de submissões públicas (`notebooks/ativo/`): melhor score
  oficial confirmado **0.893**, mantido como registro, não em
  desenvolvimento ativo (ver seção acima).

## Créditos

Ver `NOTICE.md` para a atribuição legal ao código de terceiros usado no
histórico de submissões públicas (licença Apache 2.0).
