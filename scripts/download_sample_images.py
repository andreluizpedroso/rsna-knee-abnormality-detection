"""
Baixa uma amostra mínima de imagens DICOM (só a série sagital
fluid-sensitive de N estudos com label real) pra rodar um smoke test do
pipeline (`python -m src.train --smoke-test`) sem precisar do dataset
completo (819.640 arquivos / 569.76 GB).

A Kaggle API não permite baixar uma pasta inteira nem buscar arquivos por
prefixo -- só `-f <caminho exato do arquivo>` pra download, e a listagem
(`competition_list_files`) só pagina sequencialmente (sem filtro). Por isso
este script pagina a listagem procurando os arquivos dos estudos/séries alvo
e para assim que ultrapassa (alfabeticamente) o último StudyInstanceUID
procurado -- não precisa varrer os 819 mil arquivos.

Uso:
    ./.venv/Scripts/python.exe scripts/download_sample_images.py
    ./.venv/Scripts/python.exe scripts/download_sample_images.py --n 5 --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# No Windows, importar torch depois de pandas pode quebrar o carregamento de
# DLL do torch (c10.dll) -- src.dataset importa torch, então precisa vir
# antes de qualquer import de pandas neste script.
import torch  # noqa: F401

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from src import config
from src.dataset import select_preferred_series_id

COMPETITION = "rsna-knee-abnormality-detection"
TRAIN_PREFIX = "train_series/"
CACHE_FILE = config.DATA_DIR / ".sample_download_search_cache.json"

# A conexão com kaggle.com neste ambiente tem timeouts intermitentes (visto
# várias vezes na sessão, sem relação com o conteúdo do request -- internet
# geral continua ok). Além disso, a listagem paginada bate em rate limit
# (429) com frequência quando pedimos página atrás de página rápido. Retry
# com backoff evita abortar o script por causa disso -- 429 precisa de um
# backoff bem maior que timeout de conexão comum.
MAX_RETRIES = 8
RETRY_BACKOFF_SECONDS = 5
RATE_LIMIT_BACKOFF_SECONDS = 30
PAGE_DELAY_SECONDS = 1.0  # pausa entre páginas pra reduzir a chance de 429


def with_retries(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                is_rate_limit = "429" in str(exc)
                wait = RATE_LIMIT_BACKOFF_SECONDS if is_rate_limit else RETRY_BACKOFF_SECONDS * attempt
                reason = "rate limit (429)" if is_rate_limit else "falha de conexão"
                print(f"\n  ({reason}, tentativa {attempt}/{MAX_RETRIES}, retry em {wait}s)")
                time.sleep(wait)
    raise last_exc


def select_sample_studies(n: int) -> pd.DataFrame:
    """Primeiros `n` estudos gold (label completo) por StudyInstanceUID
    ordenado -- determinístico, sem aleatoriedade -- com a série preferida
    (sagital fluid-sensitive, fallback embutido em select_preferred_series_id)
    de cada um."""
    train_df = pd.read_csv(config.TRAIN_CSV)
    series_df = pd.read_csv(config.TRAIN_SERIES_CSV)

    gold_df = train_df.dropna(subset=config.TARGET_COLUMNS)
    gold_df = gold_df.sort_values(config.ID_COLUMN).head(n)

    rows = []
    for study_id in gold_df[config.ID_COLUMN]:
        series_id = select_preferred_series_id(study_id, series_df)
        if series_id is None:
            print(f"AVISO: nenhuma série encontrada em train_series.csv para {study_id}")
            continue
        row = series_df[series_df[config.SERIES_ID_COLUMN] == series_id].iloc[0]
        rows.append({
            config.ID_COLUMN: study_id,
            config.SERIES_ID_COLUMN: series_id,
            "Anatomical_Plane": row["Anatomical_Plane"],
            "Fluid_Sensitive": row["Fluid_Sensitive"],
        })

    return pd.DataFrame(rows)


def _cache_key(targets: dict) -> str:
    return json.dumps(sorted(f"{s}/{r}" for s, r in targets))


def _load_cache(targets: dict):
    if not CACHE_FILE.exists():
        return None, None
    try:
        data = json.loads(CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None, None
    if data.get("key") != _cache_key(targets):
        return None, None  # amostra alvo mudou -- cache não serve
    loaded = {tuple(k.split("|", 1)): v for k, v in data["targets"].items()}
    return data.get("page_token"), loaded


def _save_cache(targets: dict, page_token: str | None):
    data = {
        "key": _cache_key(targets),
        "page_token": page_token,
        "targets": {f"{s}|{r}": v for (s, r), v in targets.items()},
    }
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data))


def find_target_files(api: KaggleApi, sample: pd.DataFrame, page_size: int = 200) -> dict:
    """Pagina a listagem de arquivos da competição procurando os .dcm dentro
    das pastas train_series/<StudyUID>/<SeriesUID>/ dos estudos/séries alvo.
    Para assim que a página ultrapassa (alfabeticamente) o maior StudyUID
    procurado -- os arquivos vêm ordenados por caminho, então nada depois
    disso pode mais bater. Retorna {(study_id, series_id): [caminhos .dcm]}.

    Salva progresso em CACHE_FILE a cada página -- a conexão com kaggle.com
    tem se mostrado instável (timeouts/429 intermitentes), então se o script
    cair no meio da busca, rodar de novo retoma do page_token salvo em vez de
    perder o que já foi encontrado."""
    targets = {
        (row[config.ID_COLUMN], row[config.SERIES_ID_COLUMN]): []
        for _, row in sample.iterrows()
    }
    max_study_id = max(sid for sid, _ in targets)

    page_token, cached = _load_cache(targets)
    if cached is not None:
        targets = cached
        found_total = sum(len(v) for v in targets.values())
        print(f"Retomando busca do cache: {found_total} arquivos já encontrados, continuando de onde parou.")

    page_num = 0
    while True:
        page_num += 1
        resp = with_retries(api.competition_list_files, COMPETITION, page_token=page_token, page_size=page_size)
        files = list(resp.files)
        if not files:
            break

        for f in files:
            path = f.name
            if not path.startswith(TRAIN_PREFIX):
                continue
            parts = path[len(TRAIN_PREFIX):].split("/", 2)
            if len(parts) < 3:
                continue
            study_id, series_id, _ = parts
            if study_id > max_study_id:
                print(f"\n[página {page_num}] ultrapassou o último estudo alvo ({max_study_id[:25]}...), parando a busca.")
                CACHE_FILE.unlink(missing_ok=True)
                return targets
            if (study_id, series_id) in targets:
                targets[(study_id, series_id)].append(path)

        found_total = sum(len(v) for v in targets.values())
        print(f"\r[página {page_num}] arquivos alvo encontrados até agora: {found_total}", end="", flush=True)

        page_token = resp.next_page_token
        _save_cache(targets, page_token)
        if not page_token:
            break
        time.sleep(PAGE_DELAY_SECONDS)

    print()
    CACHE_FILE.unlink(missing_ok=True)
    return targets


def download_files(api: KaggleApi, targets: dict, dry_run: bool):
    for (study_id, series_id), paths in targets.items():
        dest_dir = config.TRAIN_SERIES_DIR / study_id / series_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not paths:
            print(f"AVISO: nenhum arquivo encontrado pra {study_id[:25]}.../{series_id[:25]}...")
            continue

        already_local = {p.name for p in dest_dir.glob("*.dcm")}
        pending = [p for p in paths if Path(p).name not in already_local]

        print(f"\n{study_id[:25]}.../{series_id[:25]}... -- {len(paths)} arquivos "
              f"({len(paths) - len(pending)} já local, {len(pending)} faltando)")
        for path in pending:
            if dry_run:
                print(f"  (dry-run) {path}")
                continue
            with_retries(
                api.competition_download_file, COMPETITION, path,
                path=str(dest_dir), force=True, quiet=True,
            )
        if not dry_run:
            n_downloaded = len(list(dest_dir.glob("*.dcm")))
            print(f"  -> {n_downloaded} arquivos .dcm em {dest_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Quantos estudos gold baixar")
    parser.add_argument("--dry-run", action="store_true", help="Só lista os arquivos encontrados, não baixa")
    args = parser.parse_args()

    sample = select_sample_studies(args.n)
    print(f"\n{len(sample)} estudos selecionados:\n")
    print(sample.to_string(index=False))

    api = KaggleApi()
    api.authenticate()

    print("\n--- Procurando os arquivos das séries alvo (paginando a listagem) ---")
    targets = find_target_files(api, sample)

    print("\n--- Baixando ---")
    download_files(api, targets, args.dry_run)


if __name__ == "__main__":
    main()
