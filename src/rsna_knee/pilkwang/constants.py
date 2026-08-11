"""Constantes do pipeline público Pilkwang v15 que afetam compatibilidade.

Esses valores fazem parte do contrato dos pesos publicados: trocar slots,
pooling, grupo de slices ou agregação de TTA muda a função calculada mesmo
quando os shapes continuam aceitando o checkpoint.
"""

from __future__ import annotations

from .. import config


TARGETS = config.TARGET_COLUMNS

CROP_MM = 130.0
CACHE_IMG = 336
GROUP = 3
N_GROUP_MAX = 1
SLICE_BAND = (0.20, 0.80)
EVAL_BATCH = 8

SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]

SLOTS = SLOTS_RECOVERED
N_SLOT = len(SLOTS)

POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}

SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5),
    "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4),
    "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5),
    "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5),
    "Effusion": (0, 2),
    "Synovitis": (0, 2),
    "Baker's": (0,),
    "Contusion": (0, 1, 2),
    "Fracture": (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55

RULES_NATIVE = {
    "order": "normal",
    "lat": "centre",
    "slot_fallback": False,
    "decode_fill": "nearest",
}
RULES_LEGACY = {
    "order": "dominant_axis",
    "lat": "corner_x",
    "slot_fallback": True,
    "decode_fill": "zero",
}

TTA_OVERLAP = True
TTA_POOL = "prob"

# Patch da comunidade (renta0426): esses alvos agregam as janelas de TTA por
# máximo em vez de média, mantendo a média para os demais. Ampliado de 2 pra
# 4 targets na v15-tta-more-targets (publicScore 0.893 -> 0.897): fratura e
# ruptura de menisco (medial e lateral) e contusão costumam aparecer numa
# janela específica da pilha de slices; média dilui o sinal com janelas que
# não veem o achado, máximo captura a janela que efetivamente viu. Ampliado
# de novo pra 6 targets (0.897 -> 0.899): ACL e MCL têm o mesmo padrão
# (ligamento visível só em parte dos cortes).
TTA_TARGET_POOL = {
    "Fracture": "max",
    "Lateral Meniscus": "max",
    "Medial Meniscus": "max",
    "Contusion": "max",
    "ACL": "max",
    "MCL": "max",
}

