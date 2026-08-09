"""
Weak supervision: deriva pseudo-labels para as 12 classes a partir do texto
livre do `Report`, para os ~4.3k estudos de treino que têm laudo mas não têm
as 12 colunas de label preenchidas (ver CLAUDE.md e
notebooks/historico/01_eda.ipynb, seção 5).

Abordagem: regras/regex (estilo labeling functions), não um modelo de NLP
treinado -- os laudos vêm em vários idiomas (inglês, espanhol, alemão,
francês, português observados na amostra) e não há dados anotados
suficientes (só 58 estudos) para treinar um classificador de texto
confiável. Para cada label:

  - Labels "autoevidentes" (Effusion, Synovitis, Baker's, Contusion,
    Fracture): o próprio termo da anatomia/achado já é o achado. Positivo se
    o termo aparece sem negação próxima; negativo se aparece negado
    ("no fracture", "sin derrame", "kein Erguss"); NaN (abstenção) se o termo
    nem aparece no laudo.
  - Labels "compostas" (ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral/PF
    OA): precisam de um termo de anatomia (ex.: "menisco medial") E um termo
    de patologia próximo (ex.: "rotura", "tear") para virar positivo; termo
    de anatomia com termo de normalidade próximo ("intact", "íntegro") vira
    negativo; anatomia mencionada sem nenhum dos dois fica em NaN.

Isso é deliberadamente conservador: preferimos abster (NaN, sem pseudo-label)
a inventar um label errado. Use `evaluate_against_gold` para checar a
qualidade das regras contra os 58 estudos que já têm label verdadeiro antes
de confiar nas pseudo-labels dos demais.

O bloco de avaliação via linha de comando (antigo `if __name__ == "__main__"`
de `weak_supervision.py`) mora agora em `cli/evaluate_labels.py` -- este
módulo é só a biblioteca.

Roadmap (ver CLAUDE.md/PROGRESS.md): ponderação diferenciada gold vs.
pseudo-label por CONFIANÇA (peso por label derivado de
`evaluate_against_gold`, não um escalar único) é uma técnica futura que usa
as funções deste módulo, sem mudar a lógica de extração em si.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

from .. import config

WINDOW_CHARS = 40  # janela de contexto (chars) ao redor do termo de anatomia

# --- Termos de anatomia/achado por label (já sem acento, minúsculo) --------
ANATOMY_PATTERNS = {
    "ACL": [
        "acl", "lca", "vkb",
        "anterior cruciate ligament", "ligamento cruzado anterior",
        "ligament croise anterieur", "vorderes kreuzband", "vorderen kreuzbandes",
    ],
    "MCL": [
        "mcl", "lcm",
        "medial collateral ligament", "ligamento colateral medial",
        "ligament collateral medial", "inneres seitenband", "innenband",
        "mediales seitenband",
    ],
    "Medial Meniscus": [
        "medial meniscus", "menisco medial", "menisco interno",
        "menisque interne", "menisque medial", "innenmeniskus",
    ],
    "Lateral Meniscus": [
        "lateral meniscus", "menisco lateral", "menisco externo",
        "menisque externe", "menisque lateral", "aussenmeniskus",
    ],
    "Medial OA": [
        "medial compartment", "compartimento medial", "femorotibial medial",
        "medialen kompartiment", "compartiment medial", "medial tibiofemoral",
    ],
    "Lateral OA": [
        "lateral compartment", "compartimento lateral", "femorotibial lateral",
        "lateralen kompartiment", "compartiment lateral", "lateral tibiofemoral",
    ],
    "PF OA": [
        "patellofemoral", "femoropatelar", "femoro-patelar", "patelofemoral",
        "femoropatellar", "femoro-patellaire",
    ],
    "Effusion": [
        "effusion", "derrame", "erguss", "gelenkerguss", "epanchement",
        "joint effusion", "derrame articular",
    ],
    "Synovitis": ["synovitis", "sinovitis", "synovialitis", "synovite"],
    "Baker's": [
        "baker", "quiste de baker", "poplitealzyste", "kyste poplite",
        "popliteal cyst", "cisto de baker", "quisto de baker",
    ],
    "Contusion": [
        "contusion", "bone bruise", "contusion osea", "knochenkontusion",
        "marrow edema", "bone marrow edema", "osteochondral impaction",
        "impaction injury",
    ],
    "Fracture": ["fracture", "fractura", "fraktur", "frattura", "fratura"],
}

# Labels cujo próprio termo de anatomia já é o achado (não precisam de um
# termo de patologia separado por perto -- só checam negação).
SELF_EVIDENT_LABELS = {"Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"}

OA_LABELS = {"Medial OA", "Lateral OA", "PF OA"}

# Excluídas da geração de pseudo-labels: o gold set (58 estudos) tem pouca
# gente com essas classes preenchidas (Lateral OA n=4, PF OA n=15 em
# evaluate_against_gold), o que não dá pra calibrar/confiar na precisão das
# regras. Ficam sempre NaN em generate_pseudo_labels até termos mais sinal
# pra validar. Continuam com padrões definidos acima (não removidos) caso
# seja revisitado depois.
EXCLUDED_FROM_PSEUDO_LABELS = {"Lateral OA", "PF OA"}

OA_TERMS = [
    "osteoarthritis", "degenerative", "chondral loss", "cartilage loss",
    "osteophyte", "artrosis", "cambios degenerativos", "arthrose",
    "chondropathie", "chondropathy", "gonarthrose", "artrose",
    "joint space narrowing", "pinzamiento articular",
]

INJURY_TERMS = [
    "tear", "torn", "rupture", "ruptured", "sprain", "strain", "injury",
    "lesion", "rotura", "ruptura", "esguince", "riss", "ruptur",
    "verletzung", "dechirure", "entorse", "avulsion", "lesao",
]

NEGATIVE_TERMS = [
    "intact", "normal", "no tear", "no evidence", "without", "unremarkable",
    "within normal limits", "preserved", "no acute", "no significant",
    "sin alteraciones", "sin rotura", "sin lesion", "no hay", "ausencia",
    "ausencia de", "integro", "intacto", "intacta", "conservado", "conservada",
    "morfologia conservada", "senal conservada",
    "physiologisch", "intakt", "unauffallig", "kein", "keine", "ohne",
    "pas de", "sans", "aucun", "aucune", "normale", "integre",
]

_word = lambda term: r"\b" + re.escape(term) + r"\b"


def normalize_text(text: str) -> str:
    """minúsculo + remove acentos, pra casar termos independente de
    variação de acentuação/idioma sem precisar duplicar cada termo."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def _has_any(terms: list[str], window: str) -> bool:
    return any(re.search(_word(t), window) for t in terms)


def label_report(report_text: str) -> dict[str, float]:
    """Aplica as regras de weak supervision a um único laudo e retorna um
    dict {label: 0.0/1.0/nan}. nan = abstenção (nenhuma pista suficiente)."""
    result = {col: float("nan") for col in config.TARGET_COLUMNS}
    if not isinstance(report_text, str) or not report_text.strip():
        return result

    text = normalize_text(report_text)

    for label, anatomy_terms in ANATOMY_PATTERNS.items():
        pathology_terms = None if label in SELF_EVIDENT_LABELS else (
            OA_TERMS if label in OA_LABELS else INJURY_TERMS
        )

        votes = []
        for term in anatomy_terms:
            for m in re.finditer(_word(term), text):
                start, end = m.span()
                window = text[max(0, start - WINDOW_CHARS): end + WINDOW_CHARS]
                has_neg = _has_any(NEGATIVE_TERMS, window)

                if pathology_terms is None:
                    votes.append(0 if has_neg else 1)
                else:
                    has_pos = _has_any(pathology_terms, window)
                    if has_pos:
                        votes.append(1)
                    elif has_neg:
                        votes.append(0)
                    # nem um nem outro -> não vota (abstenção nessa ocorrência)

        if votes:
            result[label] = 1.0 if max(votes) == 1 else 0.0

    return result


def generate_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna uma cópia de `df` onde as células NaN das 12 colunas de label
    são preenchidas com pseudo-labels derivados do Report (quando há sinal
    suficiente -- ver label_report). Nunca sobrescreve um label verdadeiro já
    existente. Colunas em EXCLUDED_FROM_PSEUDO_LABELS nunca são preenchidas
    (ficam NaN). Adiciona a coluna booleana `is_pseudo_label` (True se ao
    menos uma das colunas daquela linha veio da regra, não do label
    original)."""
    out = df.copy()
    out["is_pseudo_label"] = False

    missing_mask = out[config.TARGET_COLUMNS].isna().any(axis=1) & out[config.TEXT_COLUMN].notna()
    for idx in out.index[missing_mask]:
        pseudo = label_report(out.at[idx, config.TEXT_COLUMN])
        filled_any = False
        for col, val in pseudo.items():
            if col in EXCLUDED_FROM_PSEUDO_LABELS:
                continue
            if pd.isna(out.at[idx, col]) and not np.isnan(val):
                out.at[idx, col] = val
                filled_any = True
        out.at[idx, "is_pseudo_label"] = filled_any

    return out


def evaluate_against_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Roda as regras de weak supervision nos estudos que JÁ têm as 12
    colunas de label verdadeiras (só usa o Report, ignora o label real na
    hora de rotular) e compara com o gold. Serve para calibrar a confiança
    nas regras antes de usá-las nos ~4.3k estudos sem label. Retorna um
    DataFrame com coverage/precision/recall/accuracy por label."""
    gold = df.dropna(subset=config.TARGET_COLUMNS)
    preds = pd.DataFrame(
        [label_report(r) for r in gold[config.TEXT_COLUMN]], index=gold.index
    )

    rows = []
    for col in config.TARGET_COLUMNS:
        y_true = gold[col].astype(int)
        y_pred = preds[col]
        covered = y_pred.notna()
        n = int(covered.sum())

        row = {"label": col, "coverage": covered.mean(), "n": n}
        if n == 0:
            row.update(precision=np.nan, recall=np.nan, accuracy=np.nan)
        else:
            yt = y_true[covered]
            yp = y_pred[covered].astype(int)
            tp = int(((yt == 1) & (yp == 1)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            row["precision"] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            row["recall"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            row["accuracy"] = (tp + tn) / n

        rows.append(row)

    return pd.DataFrame(rows).set_index("label")
