"""
Weak supervision: deriva pseudo-labels para as 12 classes a partir do texto
livre do `Report`, para os ~4.3k estudos de treino que têm laudo mas não têm
as 12 colunas de label preenchidas (ver CLAUDE.md e notebooks/01_eda.ipynb,
seção 5).

Abordagem: regras/regex (estilo labeling functions), não um modelo de NLP
treinado -- os laudos vêm em vários idiomas (inglês, espanhol, alemão,
francês, português observados na amostra) e não há dados anotados
suficientes (só 58 estudos) para treinar um classificador de texto
confiável. Para cada label:

  - Labels "autoevidentes com magnitude" (Effusion, Synovitis, Baker's):
    o organizador da competição publicou os critérios clínicos usados pelos
    radiologistas para montar o gold set -- só conta como positivo achado
    de magnitude MODERADA OU GRANDE; derrame/cisto/sinovite pequeno, mínimo
    ou traço é negativo (mesmo estando presente). Termo sem nenhum
    qualificador de magnitude por perto mantém o comportamento antigo
    (positivo) para não sacrificar recall à toa -- só o caso claramente
    pequeno/mínimo virou negativo.
  - Labels "autoevidente simples" (Contusion): o próprio termo já é o
    achado (edema de medula óssea por impacto).
  - Fracture: "autoevidente com gate de cronicidade" -- só conta como
    achado (a competição pede fratura AGUDA) se não houver qualificador de
    cronicidade por perto ("old fracture", "healed", "consolidada" etc.);
    caindo nesse caso vira negativo, não abstenção, porque o próprio termo
    "fracture" está presente, só que descrevendo algo antigo/consolidado.
  - Ligamentos (ACL, MCL): só positivo com termo de ruptura de ALTO GRAU
    ("complete tear", "rotura completa", "discontinuity" etc.) por perto do
    termo de anatomia; termo de gravidade leve/ambígua por perto ("sprain",
    "partial tear", "degenerative", "esguince" etc.) vira negativo (o
    critério oficial trata achado ambíguo como negativo, não como
    abstenção); nem um nem outro (só o nome do ligamento, sem qualificador
    de gravidade) fica em NaN -- não há como saber.
  - Meniscos (Medial/Lateral Meniscus): só positivo se o sinal alcança
    claramente a superfície articular ou há fragmento deslocado/truncado
    por perto do termo de anatomia; termo de degeneração intrassubstancial
    sem extensão à superfície vira negativo; nem um nem outro fica em NaN.
  - OA (Medial/Lateral/PF OA): só positivo com termo de perda de cartilagem
    de ALTO GRAU ("severe", "bone-on-bone", "grade 4" etc.) por perto;
    termo de OA leve/inicial vira negativo; menção genérica de
    "osteoarthritis"/"degenerative changes" sem grau explícito fica em NaN
    (não dá pra saber se é leve ou grave só pelo termo genérico).

Isso é deliberadamente conservador: preferimos abster (NaN, sem pseudo-label)
a inventar um label errado quando o laudo não dá indício de gravidade
nenhum. Quando o laudo DÁ um indício de gravidade (mesmo que leve/ambíguo),
seguimos o mesmo critério do gold set: ambíguo = negativo. Use
`evaluate_against_gold` para checar a qualidade das regras contra os 58
estudos que já têm label verdadeiro antes de confiar nas pseudo-labels dos
demais.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config

WINDOW_CHARS = 60  # janela de contexto (chars) ao redor do termo de anatomia --
# alargada de 40 pra 60 ao introduzir termos de gravidade multi-palavra
# (ex.: "complete discontinuity of the fibers"), que às vezes ficam um
# pouco mais longe do termo de anatomia do que os termos de negação curtos
# usados antes.

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
# termo de patologia separado por perto -- só checam negação). Effusion,
# Synovitis e Baker's também checam magnitude (ver MODERATE_LARGE_TERMS/
# SMALL_TERMS) e Fracture checa cronicidade (ver CHRONIC_TERMS) -- ambos
# tratados como uma etapa extra em cima da lógica "autoevidente" abaixo.
SELF_EVIDENT_LABELS = {"Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"}

# Dessas, exige-se magnitude MODERADA OU GRANDE pra contar como achado
# clinicamente positivo (critério oficial do gold set) -- pequeno/mínimo/
# traço é negativo mesmo estando presente.
MAGNITUDE_GATED_LABELS = {"Effusion", "Synovitis", "Baker's"}

# Só Fracture precisa do gate de cronicidade (a competição só conta fratura
# AGUDA -- uma fratura antiga/consolidada mencionada no laudo não conta).
CHRONIC_GATED_LABELS = {"Fracture"}

LIGAMENT_LABELS = {"ACL", "MCL"}
MENISCUS_LABELS = {"Medial Meniscus", "Lateral Meniscus"}
OA_LABELS = {"Medial OA", "Lateral OA", "PF OA"}

# Excluídas da geração de pseudo-labels: o gold set (58 estudos) tem pouca
# gente com essas classes preenchidas (Lateral OA n=4, PF OA n=15 em
# evaluate_against_gold), o que não dá pra calibrar/confiar na precisão das
# regras. Ficam sempre NaN em generate_pseudo_labels até termos mais sinal
# pra validar. Continuam com padrões definidos acima (não removidos) caso
# seja revisitado depois.
EXCLUDED_FROM_PSEUDO_LABELS = {"Lateral OA", "PF OA"}

# --- Termos de GRAVIDADE ALTA (só esses viram positivo) ---------------------
# Critério oficial do gold set: ACL/MCL só positivo com ruptura de alto grau
# (>50% das fibras ou descontinuidade completa); qualquer coisa mais leve é
# negativo por definição.
LIGAMENT_HIGH_GRADE_TERMS = [
    "complete tear", "full tear", "full-thickness tear", "complete rupture",
    "complete disruption", "full-thickness disruption", "high-grade tear",
    "high grade tear", "grade 3", "grade iii", "discontinuity",
    "disrupted fibers", "torn fibers", "fiber discontinuity",
    "rotura completa", "ruptura completa", "rotura total", "ruptura total",
    "discontinuidad", "alto grado", "grado alto", "grado iii",
    "ruptura de alto grado", "rotura de alto grado",
    "rupture complete", "rupture totale", "haut grade", "discontinuite",
    "kompletter riss", "vollstandiger riss", "hochgradig", "diskontinuitat",
]

# Termos de gravidade LEVE/AMBÍGUA -- por critério oficial, isso é negativo
# (não abstenção), porque o laudo já deu um indício de gravidade e esse
# indício é insuficiente pro corte oficial.
LIGAMENT_MILD_TERMS = [
    "sprain", "strain", "partial tear", "partial-thickness tear",
    "low-grade", "low grade", "grade 1", "grade i", "grade 2", "grade ii",
    "mild", "chronic", "degenerative", "degeneration", "thickening",
    "signal change", "intrasubstance", "mucoid degeneration", "chondromalacia",
    "esguince", "rotura parcial", "bajo grado", "grado i", "grado ii",
    "cronico", "cronica", "degeneracion", "engrosamiento",
    "entorse", "dechirure partielle", "bas grade", "chronique",
    "degeneratif", "epaississement",
    "teilriss", "niedriggradig", "chronisch", "degenerativ", "verdickung",
]

# Critério oficial: menisco só positivo se o sinal atinge a superfície
# articular (>=2 imagens) ou há fragmento deslocado/truncado.
MENISCUS_SURFACE_TERMS = [
    "extends to the articular surface", "extending to the articular surface",
    "extends to the surface", "extending to the surface",
    "reaches the surface", "surface extension", "articular surface",
    "displaced fragment", "displaced meniscal fragment", "truncated",
    "bucket-handle", "bucket handle", "flap tear extending",
    "extendida a la superficie articular", "extendida a la superficie",
    "superficie articular", "fragmento desplazado", "fragmento luxado",
    "asa de cubo",
    "surface articulaire", "fragment deplace", "anse de seau",
    "gelenkflache", "verlagertes fragment", "korbhenkel",
]

# Degeneração intrassubstancial / sinal sem chegar na superfície -- por
# critério oficial isso é negativo.
MENISCUS_MILD_TERMS = [
    "intrasubstance", "intrasubstance signal", "mucoid degeneration",
    "degenerative signal", "degenerative change", "signal change",
    "does not extend to the surface", "not extend to the articular surface",
    "no extend", "grade 1 signal", "grade i signal", "grade 2 signal",
    "grade ii signal",
    "degeneracion intrasustancial", "senal degenerativa", "no se extiende",
    "signal intrasubstantiel", "degenerescence mucoide",
    "intrasubstanzielles signal", "degeneratives signal",
]

# Critério oficial: OA só positivo com perda de cartilagem de alto grau
# (>50% da espessura, área >=1cm).
OA_HIGH_GRADE_TERMS = [
    "full-thickness cartilage loss", "full thickness cartilage loss",
    "high-grade cartilage loss", "high grade cartilage loss",
    "severe cartilage loss", "bone-on-bone", "bone on bone",
    "grade 4", "grade iv", "extensive cartilage loss",
    "advanced osteoarthritis", "severe osteoarthritis",
    "perdida de cartilago de alto grado", "artrosis severa",
    "artrosis avanzada", "hueso con hueso", "grado iv",
    "perte de cartilage severe", "arthrose severe", "arthrose avancee",
    "os contre os",
    "schwerer knorpelverlust", "fortgeschrittene arthrose",
    "hochgradiger knorpelverlust",
]

OA_MILD_TERMS = [
    "mild osteoarthritis", "mild degenerative", "early osteoarthritis",
    "mild cartilage loss", "chondromalacia", "small osteophyte",
    "grade 1", "grade i", "grade 2", "grade ii",
    "artrosis leve", "cambios degenerativos leves", "osteofito pequeno",
    "arthrose legere", "arthrose debutante",
    "leichte arthrose", "beginnende arthrose",
]

# --- Magnitude (Effusion/Synovitis/Baker's) ---------------------------------
MODERATE_LARGE_TERMS = [
    "moderate effusion", "large effusion", "moderate to large",
    "moderate-to-large", "large to moderate", "significant effusion",
    "moderate to severe", "moderate synovitis", "severe synovitis",
    "moderate cyst", "large cyst",
    "derrame moderado", "derrame grande", "derrame moderado a grande",
    "quiste moderado", "quiste grande",
    "epanchement modere", "epanchement important", "epanchement abondant",
    "kyste moderee", "kyste important",
    "massiger erguss", "grosser erguss", "deutlicher erguss",
]

SMALL_TERMS = [
    "trace effusion", "small effusion", "minimal effusion", "tiny effusion",
    "physiologic amount", "physiological amount", "trace synovitis",
    "mild synovitis", "minimal synovitis", "small cyst", "trace cyst",
    "derrame minimo", "derrame pequeno", "escaso derrame",
    "sinovitis leve", "sinovitis minima", "quiste pequeno",
    "epanchement minime", "epanchement discret", "epanchement physiologique",
    "synovite legere", "kyste minime",
    "geringer erguss", "minimaler erguss", "physiologische menge",
]

# --- Cronicidade (Fracture) --------------------------------------------------
CHRONIC_TERMS = [
    "old fracture", "healed fracture", "chronic fracture", "remote fracture",
    "consolidated fracture", "healing fracture", "prior fracture",
    "fractura antigua", "fractura consolidada", "fractura previa",
    "fracture ancienne", "fracture consolidee", "fracture anterieure",
    "alte fraktur", "verheilte fraktur", "konsolidierte fraktur",
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


def _vote_for_occurrence(label: str, window: str, has_neg: bool) -> int | None:
    """Decide o voto (0/1/None=abstenção nesta ocorrência) para uma menção
    do termo de anatomia de `label`, dado o texto ao redor (`window`) e se
    há negação explícita por perto. Implementa o critério oficial do gold
    set (achado ambíguo/leve = negativo, não abstenção) -- ver docstring do
    módulo para o raciocínio por tipo de label."""
    if has_neg:
        return 0

    if label in SELF_EVIDENT_LABELS:
        if label in MAGNITUDE_GATED_LABELS:
            if _has_any(MODERATE_LARGE_TERMS, window):
                return 1
            if _has_any(SMALL_TERMS, window):
                return 0
            return 1  # sem qualificador de magnitude -- mantém o comportamento antigo
        if label in CHRONIC_GATED_LABELS:
            return 0 if _has_any(CHRONIC_TERMS, window) else 1
        return 1  # self-evidente simples (Contusion)

    if label in LIGAMENT_LABELS:
        if _has_any(LIGAMENT_HIGH_GRADE_TERMS, window):
            return 1
        if _has_any(LIGAMENT_MILD_TERMS, window):
            return 0
        return None

    if label in MENISCUS_LABELS:
        if _has_any(MENISCUS_SURFACE_TERMS, window):
            return 1
        if _has_any(MENISCUS_MILD_TERMS, window):
            return 0
        return None

    if label in OA_LABELS:
        if _has_any(OA_HIGH_GRADE_TERMS, window):
            return 1
        if _has_any(OA_MILD_TERMS, window):
            return 0
        return None

    raise ValueError(f"label sem regra de voto definida: {label}")


def label_report(report_text: str) -> dict[str, float]:
    """Aplica as regras de weak supervision a um único laudo e retorna um
    dict {label: 0.0/1.0/nan}. nan = abstenção (nenhuma pista suficiente)."""
    result = {col: float("nan") for col in config.TARGET_COLUMNS}
    if not isinstance(report_text, str) or not report_text.strip():
        return result

    text = normalize_text(report_text)

    for label, anatomy_terms in ANATOMY_PATTERNS.items():
        votes = []
        for term in anatomy_terms:
            for m in re.finditer(_word(term), text):
                start, end = m.span()
                window = text[max(0, start - WINDOW_CHARS): end + WINDOW_CHARS]
                has_neg = _has_any(NEGATIVE_TERMS, window)
                vote = _vote_for_occurrence(label, window, has_neg)
                if vote is not None:
                    votes.append(vote)

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


if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_CSV)

    print("=== Avaliação das regras contra os estudos com label verdadeiro ===")
    metrics = evaluate_against_gold(df)
    print(metrics.round(3))

    print(f"\n(Excluídas das pseudo-labels: {sorted(EXCLUDED_FROM_PSEUDO_LABELS)} -- gold insuficiente pra calibrar)")

    print("\n=== Gerando pseudo-labels para os estudos sem label completo ===")
    out = generate_pseudo_labels(df)
    n_pseudo = out["is_pseudo_label"].sum()
    print(f"Estudos que ganharam ao menos 1 pseudo-label: {n_pseudo}")
    other_cols = [c for c in config.TARGET_COLUMNS if c not in EXCLUDED_FROM_PSEUDO_LABELS]
    still_missing = out.loc[out["is_pseudo_label"], other_cols].isna().any(axis=1).sum()
    print(f"Desses, ainda com NaN em alguma das 10 colunas não-excluídas: {still_missing}")
