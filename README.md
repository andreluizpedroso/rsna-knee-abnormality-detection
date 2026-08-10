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

## Abordagem atual

Há duas linhas no projeto, e essa distinção é importante:

- **Pipeline próprio/histórico** em `src/rsna_knee/`: weak supervision por
  regras sobre o `Report`, seleção de série, DINOv2-small e inferência por
  checkpoints próprios. Essa linha chegou a `publicScore ~= 0.604`.
- **Linha forte em internalização**: reprodução das ideias do notebook
  público `pilkwang/rsna-knee-baseline-v1` v15, com pesos públicos
  `pilkwang/rsna-knee-weights`, ensemble de 20 members e TTA. Essa linha
  produziu `publicScore ~= 0.893` quando rodada no Kaggle com GPU T4.

A referência pública fica em `references/pilkwang_rsna_knee_baseline_v1/`
com crédito ao autor. O código próprio de compatibilidade começou em
`src/rsna_knee/pilkwang/`; ele valida manifest/pesos, define a arquitetura
`SlotHead`/`PilkwangModel`, contratos de slots e agregação TTA por target.
Também já há o caminho de pixels/cache v15 (`walk/annotate/pick_slots/
build_cache`) e a CLI de inferência completa para rodar no Kaggle com os
pesos anexados.

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
  pilkwang/                       # contratos da linha pública v15/0.893
  cli/
    train.py                       # entrypoint: python -m src.rsna_knee.cli.train
    infer.py                       # entrypoint: python -m src.rsna_knee.cli.infer
    infer_pilkwang.py              # valida pacote de pesos públicos/TTA v15
    evaluate_labels.py             # avalia as regras de weak supervision
references/
  pilkwang_rsna_knee_baseline_v1/  # notebook público usado como referência
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

## Inferência Pilkwang v15 (`~0.893`)

Esta é a trilha que tenta reproduzir o melhor resultado conhecido do
projeto. Ela não treina: usa os pesos públicos do Pilkwang e roda somente a
inferência sobre o test set montado pelo Kaggle.

No Kaggle Notebook, anexe:

- competição `rsna-knee-abnormality-detection`;
- dataset público `pilkwang/rsna-knee-weights`;
- Kaggle Model `metaresearch/dinov2/PyTorch/small/1`;
- internet desligada para submissão;
- preferencialmente GPU T4 para reproduzir o score alto. Em CPU, o pipeline
  pode cair para perto de `0.878` por cortes de TTA/orçamento de tempo.

Dry-run para validar se os inputs foram localizados e se o `manifest.json`
dos pesos é compatível:

```bash
python -m src.rsna_knee.cli.infer_pilkwang --dry-run --out /kaggle/working/submission.csv
```

Inferência real:

```bash
python -m src.rsna_knee.cli.infer_pilkwang --out /kaggle/working/submission.csv
```

Antes de submeter, confira que `/kaggle/working/submission.csv` tem uma
linha por estudo, as 12 colunas de target, valores numéricos e nenhum nulo.
Se o score divergir muito do histórico `~0.893`, investigue primeiro:
slots/`pixel_group` do manifest, fingerprint dos checkpoints, ordenação de
slices, lateralidade (`corner_x` vs. `centre`), crop `130mm` e TTA especial
para `Fracture`/`Lateral Meniscus`.

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

O repo agora separa o pipeline próprio/histórico da referência pública que
explica o salto para `publicScore ~= 0.893`. O port modular da inferência
Pilkwang v15 está implementado em `src/rsna_knee/pilkwang/`; falta validar
no Kaggle/T4 contra o leaderboard para confirmar que ele reproduz o score
histórico.
