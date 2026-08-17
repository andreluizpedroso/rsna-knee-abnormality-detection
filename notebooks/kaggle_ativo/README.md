# Notebook real da competição (linha híbrida: RadImageNet + LLM)

Este é o snapshot versionado do notebook Kaggle que gera o melhor
`publicScore` real da competição. Substituiu a linha anterior (v15 +
TTA, `notebooks/kaggle_ativo_v15_tta/`, `publicScore = 0.899`) em
2026-08-17.

## Estado atual

- Kernel: `andreluizpedroso/rsna-knee-hybrid-radimagenet-llm`
  (https://www.kaggle.com/code/andreluizpedroso/rsna-knee-hybrid-radimagenet-llm)
- Arquivo: `rsna-knee-90-reports-llm-30-epochs.ipynb` (nome original
  mantido do notebook de origem).
- **`publicScore = 0.911` confirmado** em 2026-08-17 (submissão própria,
  rank 370 no leaderboard público no momento da confirmação, empatado
  com outros times na mesma faixa) — melhor resultado real da
  competição até agora. Histórico completo em `PROGRESS.md`.
- Origem: fork de `salemali7/rsna-knee-90-reports-llm-30-epochs`
  (notebook público, licença Apache 2.0), que por sua vez parte da linha
  do Pilkwang Kim (mesma origem da linha anterior). Soma um segundo
  backbone (ResNet50 pré-treinado em RadImageNet, além do DINOv2) e usa
  labels extraídas via LLM comercial sobre os laudos radiológicos
  (permitido pelas regras da competição, confirmado pelo host).
- **Notebook sem documentação em prosa** (ao contrário da linha anterior,
  que tinha markdown explicando cada decisão de design) — só uma célula
  com imagem. Entender o que ele faz exige ler o código diretamente.
- Score de 0.911 **não foi aceito só porque terceiros relataram** — foi
  confirmado via submissão própria antes de adotar essa linha (mesma
  metodologia usada a sessão inteira: nunca decidir por número de
  terceiro sem confirmar na nossa conta).

## Dependências Kaggle (precisa desses inputs anexados pra rodar)

Bem mais dependências que a linha anterior — **10 datasets de pelo menos
7 contas Kaggle diferentes**, não só do Pilkwang. Risco de fragilidade
real: qualquer um desses terceiros pode apagar/tornar privado o dataset
dele no futuro, quebrando a reprodução sem aviso.

- Datasets: `mattiaangeli/knee-mri-fold-weights`,
  `marwanmath/resnet-50-radimagenet-marwan`,
  `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`,
  `flight0234/rsna-knee-hybrid-report-labels`,
  `stevenleehans/rsna-knee-llm-report-labels`,
  `lixin73/rsna-knee-llm-report-labels-sol56`,
  `pilkwang/rsna-knee-llm-labels`,
  `mattiaangeli/rsna-knee-radimagenet-foldsv1-heads`,
  `pilkwang/rsna-knee-weights`, `tonylica/rsna2026-models`
- Models: `metaresearch/dinov2/PyTorch/small/1`,
  `metaresearch/dinov2/PyTorch/base/1`,
  `metaresearch/dinov2/PyTorch/large/1` (3 variantes, não só `small`)
- Competition: `rsna-knee-abnormality-detection`

Tudo isso já está registrado em `kernel-metadata.json` nesta pasta.

## Requisitos de execução

- **GPU T4** (não P100 — a API de submissão do Kaggle rejeita P100
  nesta competição. O Kaggle já reverteu o acelerador pro default (P100)
  sozinho mais de uma vez — sempre conferir/trocar antes de rodar).
- Internet Off (`enable_internet: false`, já no metadata).
- Roda bem mais pesado que a linha anterior (3 backbones DINOv2 + 1
  ResNet50, mais datasets pra baixar) — esperar mais tempo de execução.

## Checagem de segurança feita antes de adotar

Notebook escaneado por padrões de risco (`subprocess`, `os.system`,
chamadas de rede, `eval`/`exec` suspeitos) antes de rodar pela primeira
vez -- nenhum encontrado. Há um blob de dados embutido no código
(`HYB_TEACHER_PAYLOAD`, ~44KB, base64+zlib, provavelmente pseudo-labels
ou scores de um "professor" pra distillation), decodificado via
`np.load(..., allow_pickle=False)` -- seguro, essa flag bloqueia
execução de código arbitrário via pickle.

## Como iterar

Mesma lógica de antes (ver "Regra de autonomia em iteração de baixo
risco" no `CLAUDE.md`): mudanças de baixo risco são feitas direto no
notebook, não neste repo em Python separado.

1. `kaggle kernels pull andreluizpedroso/rsna-knee-hybrid-radimagenet-llm -p notebooks/kaggle_ativo -m`
2. Editar a célula relevante.
3. `kaggle kernels push -p notebooks/kaggle_ativo`
4. Rodar no Kaggle (conferir GPU T4 antes), submeter manualmente pela UI
   ("Submit to Competition" — a submissão via API tem dado erro
   persistente nesse fluxo, ver `PROGRESS.md`).
5. Se o `publicScore` melhorar: puxar de novo pra cá (`kaggle kernels
   pull`) e commitar, atualizando este README com o novo score
   confirmado. Se piorar/empatar: reverter, re-pushar, documentar a
   tentativa descartada no `PROGRESS.md`.

Esta pasta deve sempre refletir a **última config confirmada como
melhor**, nunca uma tentativa ainda pendente de resultado.
