# RSNA Knee Abnormality Detection

Modelo multimodal (MRI + laudo radiológico) para detectar 12 anormalidades
clinicamente relevantes no joelho, desenvolvido para a competição Kaggle
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

- `train.csv`: uma linha por estudo (`StudyInstanceUID`, `PatientSex`,
  `Report`, 12 colunas de label). Só uma pequena fração dos estudos de
  treino (~1.3%) tem as 12 colunas de label preenchidas — o restante vem sem
  label direto.
- `train_series.csv`: uma linha por série (`SeriesInstanceUID`,
  `Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression`) — cada estudo tem
  em média ~5.5 séries.
- `Report`: texto livre do laudo radiológico, multi-idioma, presente em
  100% dos estudos de treino. **Não existe na base de teste** — só pode ser
  usado para gerar/checar labels adicionais no treino (weak supervision),
  não como input direto do modelo em produção.
- Dataset completo: 819.640 arquivos DICOM, 569.76 GB.

## Abordagem

1. **Seleção de série**: por estudo, prioriza a série sagital
   fluid-sensitive (melhor visualização de ligamento/menisco), com fallback
   para a primeira série disponível caso a preferida não exista localmente.
2. **Weak supervision**: como só ~1.3% dos estudos de treino têm label
   completo, o texto do `Report` é usado para derivar pseudo-labels para os
   demais, via regras (não um modelo de NLP treinado — poucos exemplos
   rotulados para isso) que combinam termos de anatomia com termos de
   patologia/negação, em múltiplos idiomas. As regras são validadas contra
   os estudos com label real antes de serem aplicadas; classes sem gold
   suficiente para calibrar ficam de fora da geração de pseudo-label.
3. **Treino com peso ponderado**: pseudo-labels entram no treino com peso
   menor que labels reais no loss (ponderação por elemento, não por
   amostra inteira), refletindo a confiança menor nesse sinal. A validação
   usa exclusivamente estudos com label real.
4. **Validação estratificada multi-label**: dado o desbalanceamento entre
   classes (algumas com poucos exemplos positivos), o split usa
   `MultilabelStratifiedKFold` em vez de k-fold simples, para manter a
   proporção de cada classe em cada fold.
5. **Modelo baseline**: encoder de imagem (CNN pré-treinada) + encoder de
   texto (transformer multilíngue), com fusão por concatenação antes da
   cabeça de classificação final.

## Estrutura do projeto

```
data/                  # dados da competição (não versionado)
notebooks/
  01_eda.ipynb          # exploração inicial dos dados
src/
  config.py              # paths, colunas e hiperparâmetros centralizados
  dataset.py              # dataset multimodal + seleção de série
  model.py                 # modelo baseline (imagem + texto)
  train.py                  # loop de treino com validação estratificada
  weak_supervision.py        # geração de pseudo-labels a partir do Report
  infer.py                    # geração da submissão
scripts/
  download_data.sh            # download dos dados via Kaggle API
submissions/                   # submissões geradas localmente
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
```

Treinar localmente numa amostra pequena (validação do pipeline):

```bash
python -m src.train --smoke-test --epochs 2
```

Treinar (fold completo):

```bash
python -m src.train
```

Gerar submissão a partir de um checkpoint:

```bash
python -m src.infer --checkpoint checkpoints/best_fold0.pth
```

O treino em escala real acontece em um Kaggle Notebook (GPU gratuita, dados
já montados no ambiente) — é uma exigência da competição, já que a
submissão final não pode depender de acesso à internet nem de upload direto
de artefatos externos.

## Status

- Pipeline de dados, weak supervision e treino validados ponta a ponta.
- Treino em escala completa e submissão via Kaggle Notebook: em andamento.
