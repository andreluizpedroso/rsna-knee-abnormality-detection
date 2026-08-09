# Kaggle workflow (histórico de submissões públicas)

**⚠️ Este fluxo não é mais o pipeline ativo.** `src/rsna_knee/` voltou a
ser o pipeline em desenvolvimento (ver `CLAUDE.md`, seção "Mudança de
estratégia (2ª reversão)") -- o notebook documentado aqui foi usado
temporariamente como base de submissão e gerou o melhor score oficial
confirmado até hoje (**publicScore = 0.893**), que continua válido no
leaderboard. Este arquivo fica como referência de como reproduzir esse
resultado especificamente, caso seja útil de novo (ex.: como ponto de
comparação, ou se o roadmap de técnicas do `src/rsna_knee/` não superar
0.893 a tempo).

Este arquivo documenta, de forma objetiva, como reproduzir e submeter esse
histórico (ensemble de pesos pré-treinados públicos + patch de TTA) no
Kaggle. Ver `CLAUDE.md`/`PROGRESS.md` para o histórico completo de
decisões; este arquivo é só o "como fazer" técnico. Ver `NOTICE.md` para a
atribuição legal ao código de terceiros reaproveitado.

## Qual notebook rodar (histórico -- não é o pipeline ativo hoje)

**`notebooks/ativo/04_ensemble_infer/rsna-knee-ensemble-infer.ipynb`**
(kernel `andreluizpedroso/rsna-knee-ensemble-infer` no Kaggle) — melhor
score oficial confirmado: **publicScore = 0.893** (macro AUC-ROC).

Este notebook não treina do zero: carrega os 20 members de um pacote de
pesos já publicado (`infer_from_package`) e faz TTA por member, com
pooling por target (`TTA_TARGET_POOL`) — ajuste incorporado de um fork da
comunidade sobre o notebook original (ver `NOTICE.md`). Rodar do zero (sem
GPU T4, ver abaixo) leva a resultado pior (0.878, testado em CPU) — não
porque o modelo mude, mas porque o orçamento de tempo (`TIME_BUDGET`)
força o notebook a descartar janelas de TTA/members quando a máquina é
mais lenta.

Para atualizar o `.ipynb` local depois de uma mudança feita direto no
Kaggle:

```bash
export PYTHONIOENCODING=utf-8   # evita crash de encoding no Windows
kaggle kernels pull andreluizpedroso/rsna-knee-ensemble-infer \
  -p notebooks/ativo/04_ensemble_infer -m
```

**⚠️ Este pull traz o conteúdo de volta em inglês.** Os comentários/
docstrings/markdown do notebook foram traduzidos pra português (decisão
do usuário, aceitando abrir mão da fidelidade byte-a-byte com o pull
original -- ver `CLAUDE.md`). Depois de rodar o pull acima pra capturar
uma mudança real de código/config, é preciso traduzir de novo antes de
commitar -- não é seguro só sobrescrever com o pull cru.

## Datasets/models que precisam estar anexados

Ver `notebooks/ativo/04_ensemble_infer/kernel-metadata.json` para a lista exata
(`dataset_sources`/`model_sources`) — reproduzida aqui pra referência
rápida. Os três primeiros são datasets **de terceiros** (não nossos —
não renomeáveis, exigidos como estão pra reproduzir o resultado; ver
`NOTICE.md`):

- pacote de assets do notebook original (figuras/exemplos).
- pacote de labels/pseudo-labels publicados.
- pacote de pesos: os 20 members treinados (4 seeds × 5 folds),
  `manifest.json` + um `.pt` por member (~89MB cada, ~1.8GB total).
- `andreluizpedroso/rsna-knee-offline-wheels` — `pydicom` empacotado
  offline (não vem pré-instalado no ambiente Kaggle e o kernel roda sem
  internet). Este é nosso.
- Model: `metaresearch/dinov2/PyTorch/small/1`.

`enable_internet: false` no `kernel-metadata.json` — é uma exigência da
competição (code competition, sem acesso à internet na submissão), então
qualquer dependência tem que já estar pré-instalada no ambiente Kaggle ou
vir de um dataset anexado como wheel offline.

## Quando selecionar GPU T4 manualmente

**Sempre que o objetivo for reproduzir o score 0.893 (não o 0.878 de
CPU).** A API do Kaggle (`kagglesdk`/CLI) não tem campo para escolher o
*tipo* de GPU — só liga/desliga GPU (`enable_gpu: true/false`). O tipo
(T4 x2, P100, etc.) só é selecionável na UI web, em **Notebook Options**.

Duas restrições reais que tornam isso obrigatório, não só uma otimização:

1. **P100 é banido pela competição.** Uma submissão gerada por um kernel
   que rodou em P100 é rejeitada com HTTP 400:
   `"Submission not allowed: Your Notebook cannot use P100 GPUs in this
   competition."` — esse erro só aparece chamando
   `kaggle.api.competition_submit_cli(...)` direto em Python e capturando
   `requests.exceptions.HTTPError` (`e.response.text`); o CLI (`kaggle
   competitions submit`) engole a mensagem real e só mostra `400 Client
   Error: Bad Request`.
2. **Kaggle às vezes seleciona P100 por padrão**, mesmo depois do usuário
   reportar ter trocado pra T4 na UI — já aconteceu mais de uma vez nesta
   sessão (inclusive rodando o kernel inteiro até o fim em P100 antes de
   descobrir, na hora de submeter). **Sempre confirmar visualmente na UI
   que "GPU T4 x2" está selecionado ANTES de clicar em "Save & Run All"**,
   não confiar em memória de configuração anterior — e conferir DE NOVO
   depois de completar, comparando o `submission.csv` baixado com a
   tentativa anterior (se forem idênticos, o "Save & Run All" pode não
   ter disparado uma run nova de fato).

Fluxo recomendado:

```bash
export PYTHONIOENCODING=utf-8
kaggle kernels push -p notebooks/ativo/04_ensemble_infer   # dispara uma run
```

1. Abrir o kernel na UI do Kaggle.
2. Nas opções do notebook (ícone de engrenagem/painel lateral), conferir
   **Accelerator → GPU T4 x2** (não P100).
3. Clicar em **Save & Run All** (a run disparada pelo `push` via API pode
   já ter começado em outra config — reiniciar manualmente na UI garante
   T4).
4. Acompanhar via `kaggle kernels status <kernel-ref>` até `COMPLETE`.

**Cuidado com quota de sessão GPU**: a conta tem um limite de sessões GPU
simultâneas (`"Maximum batch GPU session count of 2 reached"` se
exceder). Um `kaggle kernels push` em um kernel com `enable_gpu: true`
**sempre dispara uma run nova**, mesmo que a intenção fosse só
renomear/ajustar metadata — não existe um modo "salvar sem rodar" via
API. Evitar pushes desnecessários em kernels GPU quando só a metadata
precisa mudar.

## Submissão oficial

```bash
export PYTHONIOENCODING=utf-8
kaggle kernels output andreluizpedroso/rsna-knee-ensemble-infer \
  -p <pasta-scratch>

kaggle kernels status andreluizpedroso/rsna-knee-ensemble-infer
# anotar o número da versão que rodou (ex.: v3)
```

Se `kaggle competitions submit` via CLI falhar com um 400 sem detalhe
(`Bad Request`), rodar via Python direto pra ver a mensagem real:

```python
import kaggle, requests
kaggle.api.authenticate()
try:
    kaggle.api.competition_submit_cli(
        file_name="<caminho>/submission.csv",
        message="<descrição>",
        competition="rsna-knee-abnormality-detection",
        kernel="andreluizpedroso/rsna-knee-ensemble-infer",
        version=<N>,
    )
except requests.exceptions.HTTPError as e:
    print(e.response.status_code, e.response.text)
```

Limite: **5 submissões/dia**, 2 finais selecionáveis pro julgamento.

## Gotcha: Kaggle remove o "+" de nomes de arquivo em datasets

Ao subir um wheel como `torch-2.5.1+cu118-cp311-...whl` como dataset,
o Kaggle silenciosamente remove o `+` do nome
(`torch-2.5.1cu118-cp311-...whl`), o que quebra o nome em uma versão
PEP440 inválida — `pip install` rejeita mesmo instalando por caminho
direto (`ERROR: Invalid requirement: 'torch==2.5.1cu118'`).

**Contorno**: em tempo de execução, copiar o wheel pra um diretório
gravável reinserindo o `+` via regex antes de instalar:

```python
import re, shutil
from pathlib import Path

def fix_wheel_name(src: Path, dst_dir: Path) -> Path:
    fixed = re.sub(r"(\d)(cu\d+)", r"\1+\2", src.name)
    dst = dst_dir / fixed
    shutil.copy(src, dst)
    return dst
```

Isso só é necessário pro dataset de wheels do torch/cu118
(`andreluizpedroso/rsna-knee-offline-torch-cu118`), usado pelas variantes
mais antigas do pipeline (cópia direta do notebook original, split
treino/infer). O `04_ensemble_infer` atual usa apenas
`andreluizpedroso/rsna-knee-offline-wheels` (pydicom), que não tem esse
problema.

## O que não vale a pena mexer sem motivo forte

- **`src/rsna_knee/data/labels.py`** (ou qualquer refinamento de regras de
  extração do `Report`): o host confirmou que o gabarito (labels "gold")
  vem sempre de leitura de imagem por radiologistas MSK, nunca do texto do
  laudo — extração de texto tem um teto embutido, por melhor que fique.
  Já testamos revisar as regras uma vez ("ambíguo=negativo") e o resultado
  real piorou (0.577→0.538). Ver `CLAUDE.md`, seção "gabarito é sempre da
  IMAGEM".
- **Trocar DINOv2 por DINOv3**: já testado em comparação direta no mesmo
  pipeline e piorou (0.775→0.763). Não vale trocar backbone só por ser
  "mais novo".
- **Decidir entre variantes usando só os 58 estudos "gold"**: desvio-padrão
  de comparação ≈0.0125 nessa amostra pequena e enviesada (positivos
  enriquecidos ~1.5x, até 3.1x em Fracture) — só diferenças >~0.02 são
  confiáveis. Usar o holdout maior (~880 estudos, baseado nos 4.407
  laudos completos com pseudo-label) como sinal principal; os 58 gold só
  pra calibração de threshold/checagem de sanidade. Ver `CLAUDE.md`, seção
  "Metodologia de validação local".
- **Comparar holdout interno de um member individual contra o valor
  gravado no manifest sem ressalva**: o script original usado pra gerar os
  20 members (4 seeds × 5 folds) nunca foi publicado — qualquer reprodução
  de split (`grp == fold`) é uma suposição razoável mas não verificada.
  Uma divergência entre o holdout medido localmente e o gravado no
  manifest pode ser diferença de composição do conjunto, não só ruído
  amostral. Ver `CLAUDE.md`, seção "Investigação: divergência de holdout".

## Scores confirmados (histórico rápido)

| Notebook / abordagem | publicScore |
|---|---|
| `src/` (pipeline histórico, DINOv2-small + lateralidade) | 0.604 |
| Cópia direta do notebook original (treina do zero, T4) | 0.824 |
| Split treino/infer (rank-mean r224+r336) | 0.833 |
| Ensemble + patch TTA, CPU | 0.878 |
| Ensemble + patch TTA, GPU T4 (**atual melhor**) | **0.893** |

Detalhe completo de cada linha (datas, kernels, decisões) em `PROGRESS.md`.

## Identidade do projeto: kernels/datasets renomeados no Kaggle

O projeto não deve ficar vinculado ao nome do autor do notebook original
na sua estrutura (nomes de kernel/dataset, pastas, títulos de seção) —
só o `NOTICE.md` traz esse nome, como exigência legal da licença Apache
2.0. Nomes de kernel/dataset são renomeáveis via API mantendo o mesmo
`id_no` (identificador numérico imutável) e trocando só o campo `slug`;
isso preserva o histórico de versões (confirmado empiricamente: um
push com `id_no` inalterado e `id`/slug novo incrementa a versão do
MESMO kernel, não cria um novo — e referências `kernel_sources` de outros
kernels nossos resolvem automaticamente pro novo slug).

Estado atual (ver `PROGRESS.md` pra data/detalhes de cada rename):

| Antes | Depois | Status |
|---|---|---|
| `rsna-knee-pilkwang-train` | `rsna-knee-legacy-split-train` | ✅ renomeado |
| `rsna-knee-pilkwang-infer` | `rsna-knee-legacy-split-infer` | ✅ renomeado |
| `rsna-knee-pilkwang-v15-infer` (kernel ATIVO) | `rsna-knee-ensemble-infer` | ⏳ pendente — bloqueado por quota de sessão GPU (`Maximum batch GPU session count of 2 reached`, ver seção acima); local (`notebooks/ativo/04_ensemble_infer/`) já reflete o nome novo, falta só reenviar (`kaggle kernels push`) quando houver vaga de sessão livre |
| `rsna-knee-v15-infer-patched-f335` | `rsna-knee-ensemble-infer-patched-f335` | ⏳ pendente — mesma razão; adiado até resolver a submissão de teste em andamento (ver `PROGRESS.md`) |
| dataset `rsna-knee-pilkwang-offline-torch-cu118` | `rsna-knee-offline-torch-cu118` | ⏳ pendente — a API de datasets do Kaggle não expõe um mecanismo de rename equivalente ao de kernels (sem campo tipo `id_no` separado do slug no fluxo de versão); a única forma confirmada é criar um dataset novo e migrar o conteúdo. Baixa prioridade: só é referenciado por kernels legados (cópia direta + split treino/infer), não pelo pipeline ativo. |
| `rsna-knee-baseline-v1-copy` | (mantido) | não renomeado — o nome não contém o nome do autor original (é o título genérico do notebook, "baseline-v1"), fora do escopo estrito do pedido; avaliar depois se vale renomear também |

**Nota sobre custo**: renomear um kernel com `enable_gpu: true` via `push`
dispara uma run nova (não existe "salvar metadata sem rodar" na API) — os
dois renames de kernels CPU-leves acima já consumiram 2 das sessões GPU
simultâneas da conta sem necessidade real de reprocessamento. Planejar
renames de kernels GPU-pesados (o kernel ativo, o de teste de member
trocado) para um momento sem outras rodadas GPU concorrentes pendentes.
