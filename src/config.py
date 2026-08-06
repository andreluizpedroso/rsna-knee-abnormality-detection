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

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_SERIES_CSV = DATA_DIR / "train_series.csv"
TEST_CSV = DATA_DIR / "test.csv"
TEST_SERIES_CSV = DATA_DIR / "test_series.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

TRAIN_SERIES_DIR = DATA_DIR / "train_series"  # <StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
TEST_SERIES_DIR = DATA_DIR / "test_series"

SUBMISSIONS_DIR = ROOT_DIR / "submissions"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

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
IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 10
IMAGE_BACKBONE = "resnet50"   # timm model name; trocar por algo maior depois
TEXT_MODEL = "distilbert-base-multilingual-cased"  # laudos vêm em vários idiomas

# Peso no BCE loss dos exemplos com pseudo-label (weak supervision via
# src/weak_supervision.py) em relação aos 58 exemplos com label real (peso
# 1.0). Precisão das regras ficou em ~0.5-0.8 contra o gold set (ver
# evaluate_against_gold) -- peso reduzido evita que o ruído deles domine o
# gradiente. É por-elemento (por estudo x label), não só por-amostra: uma
# célula sem pseudo-label (regra abstev) tem peso 0.
PSEUDO_LABEL_WEIGHT = 0.3

# Quantos slices amostrar por série ao montar o exemplo de treino (ver
# dataset.py) -- estratégia inicial simples: pega o slice do meio de cada
# série selecionada. Revisitar depois (MIL / 2.5D / 3D) para melhor sinal.
SLICES_PER_SERIES = 1
