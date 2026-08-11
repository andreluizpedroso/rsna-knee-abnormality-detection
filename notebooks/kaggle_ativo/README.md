# Notebook real da competição (linha Pilkwang)

Este é o snapshot versionado do notebook Kaggle que gera o melhor
`publicScore` real da competição. Antes de existir aqui, ele só existia
nos servidores do Kaggle e numa pasta temporária do sistema (fora do
git) — puxado/editado/empurrado via `kaggle kernels pull`/`push` direto
na conta do Kaggle. Essa pasta existe pra não depender só disso.

## Estado atual

- Kernel: `andreluizpedroso/rsna-knee-v15-tta-more-targets`
  (https://www.kaggle.com/code/andreluizpedroso/rsna-knee-v15-tta-more-targets)
- Arquivo: `rsna-knee-v15-tta-more-targets.ipynb` (código real, célula
  `TTA_TARGET_POOL` é o que muda a cada iteração — ver "Como iterar"
  abaixo).
- **`publicScore = 0.899` confirmado** em 2026-08-10, com
  `TTA_TARGET_POOL` = `"max"` em 6 targets: `Fracture`,
  `Lateral Meniscus`, `Medial Meniscus`, `Contusion`, `ACL`, `MCL`.
  Esse é o melhor resultado real da competição até agora (histórico
  completo em `PROGRESS.md`, seção "linha Pilkwang").
- Esse notebook é a versão modificada em cima do fork da comunidade
  (renta0426, publicScore 0.893), que por sua vez roda o pacote de pesos
  públicos `pilkwang/rsna-knee-weights` (20 members, ensemble) sobre a
  arquitetura DINOv2-small com atenção por slot.

## Dependências Kaggle (precisa desses inputs anexados pra rodar)

- Datasets: `pilkwang/pilkwang-public-dataset-for-notebooks-figures`,
  `pilkwang/rsna-knee-llm-labels`, `pilkwang/rsna-knee-weights`,
  `andreluizpedroso/rsna-knee-offline-wheels`
- Model: `metaresearch/dinov2/PyTorch/small/1`
- Competition: `rsna-knee-abnormality-detection`

Tudo isso já está registrado em `kernel-metadata.json` nesta pasta.

## Requisitos de execução

- **GPU T4** (não P100 — a API de submissão do Kaggle rejeita P100
  nesta competição com `"Your Notebook cannot use P100 GPUs in this
  competition"`. O Kaggle já reverteu o acelerador pro default (P100)
  sozinho mais de uma vez — sempre conferir/trocar antes de rodar).
- Internet Off (`enable_internet: false`, já no metadata).

## Como iterar (ver também a "Regra de autonomia em iteração de baixo
risco" no `CLAUDE.md`)

Mudanças de baixo risco (trocar modo de pooling do TTA, adicionar/tirar
target de `TTA_TARGET_POOL`, etc.) são feitas direto no notebook, não
neste repo em Python separado:

1. `kaggle kernels pull andreluizpedroso/rsna-knee-v15-tta-more-targets -p notebooks/kaggle_ativo -m`
2. Editar a célula que define `TTA_TARGET_POOL` (hoje é a célula de
   índice 31 no `.ipynb`, mas o índice pode mudar se outras células
   forem adicionadas — buscar por `TTA_TARGET_POOL = {`).
3. `kaggle kernels push -p notebooks/kaggle_ativo`
4. Rodar no Kaggle (conferir GPU T4 antes), submeter manualmente pela UI
   ("Submit to Competition" — a submissão via API tem dado erro
   persistente nesse fluxo, ver `PROGRESS.md`).
5. Se o `publicScore` melhorar: puxar de novo pra cá (`kaggle kernels
   pull`) e commitar, atualizando este README com o novo score
   confirmado. Se piorar/empatar: reverter a célula pra última config
   confirmada, re-pushar, documentar a tentativa descartada no
   `PROGRESS.md`.

Esta pasta deve sempre refletir a **última config confirmada como
melhor**, nunca uma tentativa ainda pendente de resultado.
