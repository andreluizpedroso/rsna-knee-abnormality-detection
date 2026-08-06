#!/usr/bin/env bash
# Baixa os dados da competição RSNA Knee Abnormality Detection.
#
# ATENÇÃO: o dataset completo tem 569.76 GB (819.640 arquivos DICOM) --
# não cabe/não faz sentido baixar tudo numa máquina comum. Por padrão este
# script baixa só os CSVs (metadados/labels/reports), que já são suficientes
# para a EDA e para prototipar o pipeline.
#
# Para baixar tudo (só se você realmente tiver espaço e motivo p/ isso):
#   bash scripts/download_data.sh --full
#
# Pré-requisitos:
#   1. pip install kaggle
#   2. Seu kaggle.json em ~/.kaggle/kaggle.json (chmod 600)
#   3. Ter clicado em "Join Competition" e aceitado as regras no site do Kaggle
set -euo pipefail

COMPETITION="rsna-knee-abnormality-detection"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DEST_DIR"

if [[ "${1:-}" == "--full" ]]; then
    echo "Baixando o dataset COMPLETO (569.76 GB) para $DEST_DIR ..."
    kaggle competitions download -c "$COMPETITION" -p "$DEST_DIR"
    ZIP_FILE="$DEST_DIR/${COMPETITION}.zip"
    if [ -f "$ZIP_FILE" ]; then
        echo "Extraindo..."
        unzip -q -o "$ZIP_FILE" -d "$DEST_DIR"
        rm "$ZIP_FILE"
    fi
else
    echo "Baixando só os arquivos CSV (metadados/labels/reports) para $DEST_DIR ..."
    for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
        echo "  -> $f"
        kaggle competitions download -c "$COMPETITION" -f "$f" -p "$DEST_DIR" --force
    done
    echo "Para baixar as imagens DICOM também, rode: bash scripts/download_data.sh --full"
    echo "(ou baixe só uma pasta específica com a Kaggle API/CLI para prototipar antes de ir pro Kaggle Notebook)"
fi

echo "Pronto. Conteúdo de $DEST_DIR:"
ls -la "$DEST_DIR"
