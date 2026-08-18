# Notebook real da competição (linha ativa: DINO-RadImageNet Rank Ensemble)

Este é o snapshot versionado do notebook Kaggle que gera o melhor
`publicScore` real da competição. Substituiu a linha anterior (híbrida
RadImageNet+LLM, `notebooks/kaggle_ativo_hybrid_radimagenet_llm/`,
`publicScore = 0.911`) em 2026-08-17.

## Estado atual

- Kernel: `andreluizpedroso/rsna-knee-tony-rank-ensemble`
  (https://www.kaggle.com/code/andreluizpedroso/rsna-knee-tony-rank-ensemble)
- Arquivo: `rsna-knee-dino-radimagenet-rank-ensemble.ipynb`.
- **`publicScore = 0.920` confirmado** em 2026-08-17 21:38:15 (submissão
  própria) — melhor resultado real da competição até agora. Histórico
  completo em `PROGRESS.md`.
- Origem: baseado no dataset de assets `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`.
  Ensemble de 35 checkpoints + 1 encoder compartilhado:
  - 20 checkpoints DINOv2-small (5 folds)
  - 5 checkpoints DINOv3-small (5 folds) — pesos próprios embutidos no
    dataset de assets, não são um `model_source` externo do Kaggle
  - 5 cabeças de atenção RadImageNet (referência) + 5 cabeças E13
    RadImageNet, reaproveitadas numa segunda disposição de slots de
    imagem (view extra, não checkpoints adicionais)
  - 1 encoder ResNet-50 RadImageNet compartilhado pelas 10 cabeças
  - Combinação final: rank ensemble (não é média direta de
    probabilidade — evita que um arm com escala/calibração diferente
    domine o blend)
- Bem menos dependências externas que a linha híbrida anterior (10
  datasets/7 contas) — aqui é essencialmente 1 dataset de assets
  (`tonylica`) + o encoder DINOv2-small público. Superfície de risco de
  fragilidade de terceiros bem menor.
- **Notebook sem documentação em prosa** além da célula inicial de
  inventário dos models — entender qualquer célula exige ler o código.

## Dependências Kaggle (precisa desses inputs anexados pra rodar)

- Datasets: `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`
- Models: `metaresearch/dinov2/PyTorch/small/1`
- Competição: `rsna-knee-abnormality-detection`
- GPU T4 habilitada, sem internet (padrão de code competition)

## Como iterar

1. `kaggle kernels pull andreluizpedroso/rsna-knee-tony-rank-ensemble -p notebooks/kaggle_ativo -m`
2. Editar a célula relevante direto no `.ipynb` (4 células: markdown de
   inventário, arm DINOv2/DINOv3, arm RadImageNet timm/OpenCV, arm
   RadImageNet ResNet-50).
3. `kaggle kernels push -p notebooks/kaggle_ativo` — **conferir GPU T4**
   antes de rodar (Kaggle já reverteu pro P100 default outras vezes
   nesta competição).
4. Rodar no Kaggle, submeter manualmente pela UI.
5. Se `publicScore` melhorar: `kaggle kernels pull` de novo, commitar,
   atualizar este README e `PROGRESS.md`. Se piorar/empatar: reverter,
   re-pushar, documentar a tentativa descartada.
