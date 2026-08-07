"""
Configuração centralizada do projeto.

Estrutura confirmada na aba Data do Kaggle (não são mais suposições):

train.csv (uma linha por STUDY):
    StudyInstanceUID, PatientSex (Male/Female, pode ser vazio),
    Report (texto livre do laudo, multi-idioma),
    + 12 colunas de label binário (0/1) -- MAS só uma pequena parte dos
    estudos de treino tem essas 12 colunas preenchidas. O resto vem sem
    label e o Report é fornecido para você tentar derivar labels (weak
    supervision / NLP sobre o laudo).

train_series.csv (uma linha por SÉRIE; cada estudo tem várias séries):
    StudyInstanceUID, SeriesInstanceUID,
    Fluid_Sensitive (0/1), Fat_Suppression (0/1),
    Anatomical_Plane (Sagittal/Coronal/Axial)

train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
    Uma série = uma sequência de MRI, com 20-45 slices (mediana 30, cauda
    longa até algumas centenas). Cada estudo tem múltiplas séries.

test.csv: StudyInstanceUID apenas (arquivo de exemplo com 3 IDs; no scoring
real vira ~1300 estudos). test_series.csv/test_series/ mesmo esquema do
treino.

sample_submission.csv: StudyInstanceUID + 12 colunas, tudo com 0.5.

Tamanho: 819.640 arquivos, 569.76 GB. NÃO dá pra baixar tudo local numa
máquina comum -- ver scripts/download_data.sh (baixa só os CSVs por
padrão) e README.md.

Não há coluna de paciente (PatientID) -- cada linha de train.csv já é um
estudo único, então dá pra usar (Multilabel)StratifiedKFold direto em
StudyInstanceUID, sem risco de vazamento entre folds.
"""

from pathlib import Path

COMPETITION_SLUG = "rsna-knee-abnormality-detection"

# --- Paths -------------------------------------------------------------
# Kaggle Notebooks sempre têm /kaggle/input montado (mesmo sem internet) --
# é o jeito mais confiável de detectar o ambiente sem depender de variável
# de ambiente específica. Localmente, os dados vêm de scripts/download_data.sh
# em <raiz do repo>/data/; no Kaggle, a competição já vem montada em
# /kaggle/input/<slug>/ e só /kaggle/working/ é gravável.
IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    # O caminho de montagem varia conforme como a competição foi anexada:
    # kernels criados via API (kagglesdk) montam em
    # /kaggle/input/competitions/<slug>/, já o fluxo clássico "Add Data" pela
    # UI monta direto em /kaggle/input/<slug>/. Testa os candidatos nessa
    # ordem e usa o primeiro que já tiver train.csv; se nenhum tiver (ainda
    # não anexado), cai no path clássico -- o erro de arquivo não encontrado
    # fica claro de qualquer forma.
    _KAGGLE_DATA_CANDIDATES = [
        Path("/kaggle/input/competitions") / COMPETITION_SLUG,
        Path("/kaggle/input") / COMPETITION_SLUG,
    ]
    DATA_DIR = next(
        (p for p in _KAGGLE_DATA_CANDIDATES if (p / "train.csv").exists()),
        _KAGGLE_DATA_CANDIDATES[-1],
    )
    WORKING_DIR = Path("/kaggle/working")
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    WORKING_DIR = ROOT_DIR

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_SERIES_CSV = DATA_DIR / "train_series.csv"
TEST_CSV = DATA_DIR / "test.csv"
TEST_SERIES_CSV = DATA_DIR / "test_series.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

TRAIN_SERIES_DIR = DATA_DIR / "train_series"  # <StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
TEST_SERIES_DIR = DATA_DIR / "test_series"

SUBMISSIONS_DIR = WORKING_DIR / "submissions"
CHECKPOINT_DIR = WORKING_DIR / "checkpoints"

# --- Labels --------------------------------------------------------------
TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
ID_COLUMN = "StudyInstanceUID"
SERIES_ID_COLUMN = "SeriesInstanceUID"
TEXT_COLUMN = "Report"
SEX_COLUMN = "PatientSex"

# --- Treino ----------------------------------------------------------------
SEED = 42
# 3 (não 5) -- só 58 estudos com label completo e MCL tem 9 positivos; com
# 5 folds um fold de validação ficaria com ~11-12 exemplos e risco real de
# não ter nenhum positivo de MCL. MultilabelStratifiedKFold (iterstrat) é
# usado em vez de KFold/StratifiedKFold comum porque o problema é multi-label.
N_FOLDS = 3
# 392 (não 384) -- múltiplo de 14, exigido pelo patch size do backbone
# DINOv2 (ViT-S/14) abaixo. Se algum dia voltar pra um backbone CNN
# (resnet etc.), qualquer tamanho serve; 384 era só o valor usado antes.
IMAGE_SIZE = 392
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 10

# DINOv2-small (ver PROGRESS.md, seção DINOv2): ViT auto-supervisionado da
# Meta, pré-treinado em bilhões de imagens sem label -- costuma superar
# CNNs clássicas quando há poucos exemplos rotulados pra fine-tuning,
# exatamente nosso caso (só 58 gold). Vários notebooks públicos da
# competição usando DINOv2 reportam publicScore 0.77-0.81, bem acima do
# baseline resnet50 anterior (0.577). Escolhido "small" (~22M parâmetros)
# em vez de "base" (~86M) como primeiro teste -- mais rápido de
# treinar/rodar no Kaggle, valida mais rápido se a troca de backbone ajuda
# antes de investir numa rodada maior de GPU no "base". O timm já registra
# os pesos DINOv2 (baixa do HF Hub igual ao resnet50 antes -- não precisa
# de torch.hub nem de Kaggle Model separado). model.py detecta "dinov2" no
# nome do backbone e ativa dynamic_img_size automaticamente (exigido pra
# aceitar IMAGE_SIZE=392 em vez do tamanho de pré-treino padrão, 518).
IMAGE_BACKBONE = "vit_small_patch14_dinov2.lvd142m"

# Peso no BCE loss dos exemplos com pseudo-label (weak supervision via
# src/weak_supervision.py) em relação aos 58 exemplos com label real (peso
# 1.0). Precisão das regras ficou em ~0.5-0.8 contra o gold set (ver
# evaluate_against_gold) -- peso reduzido evita que o ruído deles domine o
# gradiente. É por-elemento (por estudo x label), não só por-amostra: uma
# célula sem pseudo-label (regra abstev) tem peso 0.
PSEUDO_LABEL_WEIGHT = 0.3

# Nº de épocas sem melhora de val_auc antes de parar o treino do fold (early
# stopping) -- o treino do fold 0 (10 épocas, sem early stopping) mostrou
# overfitting claro depois da época 4 (val_auc 0.6085 -> 0.5588 na época 10).
EARLY_STOPPING_PATIENCE = 3

# Quantos slices amostrar por série ao montar o exemplo de treino (ver
# dataset.py) -- estratégia inicial simples: pega o slice do meio de cada
# série selecionada. Revisitar depois (MIL / 2.5D / 3D) para melhor sinal.
SLICES_PER_SERIES = 1
