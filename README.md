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

## Abordagem

Pipeline próprio em `src/rsna_knee/`: weak supervision por regras sobre o
`Report`, seleção de série sagital fluid-sensitive, normalização de
lateralidade via geometria DICOM, encoder DINOv2-small + cabeça de
classificação para as 12 classes. Roadmap de técnicas a implementar
(ordenação de slice por geometria, TTA, ensemble ponderado por holdout,
etc.) documentado no `CLAUDE.md`.

## Estrutura do projeto

```
data/                          # dados da competição (não versionado)
notebooks/historico/
  01_eda.ipynb                   # exploração inicial dos dados
  02_train_kaggle.ipynb          # treino no Kaggle (Internet On, GPU)
  03_submit_kaggle.ipynb         # submissão no Kaggle (Internet Off)
src/rsna_knee/                  # pacote do pipeline
  config.py                      # paths, hiperparâmetros, lista de targets
  data/
    dicom.py                       # I/O de baixo nível (leitura, normalização, resize)
    laterality.py                  # normalização de lateralidade via geometria DICOM
    series.py                      # seleção de série (sagital fluid-sensitive)
    slices.py                      # seleção de slice dentro da série
    labels.py                      # weak supervision (pseudo-labels a partir do Report)
    dataset.py                     # Dataset/DataLoader do PyTorch
  modeling/
    backbone.py                    # encoder de imagem (DINOv2 via timm)
    heads.py                       # cabeça de classificação
    model.py                       # composição backbone + head
    ensemble.py                    # ensemble entre checkpoints (roadmap, vazio hoje)
  training/
    validation.py                  # AUC-ROC macro
    loop.py                        # loop de treino/validação por fold
  inference/
    submission.py                  # carrega checkpoints, ensemble, gera submission.csv
    tta.py                         # TTA de slice (roadmap, vazio hoje)
  cli/
    train.py                       # entrypoint: python -m src.rsna_knee.cli.train
    infer.py                       # entrypoint: python -m src.rsna_knee.cli.infer
    evaluate_labels.py             # avalia as regras de weak supervision
tests/                          # pytest, sem GPU/dados reais (sintéticos/mocks)
scripts/
  download_data.sh                # download dos dados via Kaggle API
  download_sample_images.py       # baixa amostra mínima pra --smoke-test
checkpoints/                    # saída local do treino (*.pth gitignored, nunca vai pro GitHub)
submissions/                    # submissões geradas localmente
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
já montados no ambiente) via `notebooks/historico/02_train_kaggle.ipynb` —
é uma exigência da competição, já que a submissão final
(`03_submit_kaggle.ipynb`) não pode depender de acesso à internet nem de
upload direto de artefatos externos.

## Status

Pipeline (`src/rsna_knee/`) reestruturado em pacote modular, paridade de
comportamento confirmada contra a versão anterior. Melhor score oficial
confirmado até agora: 0.604 (macro AUC-ROC, publicScore) — roadmap de
técnicas novas priorizado em `CLAUDE.md`.
