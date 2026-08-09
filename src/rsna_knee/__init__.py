"""Pacote do pipeline ativo do projeto RSNA Knee Abnormality Detection.

Reestruturação (a partir dos módulos soltos originais em `src/*.py`) em
subpacotes de responsabilidade única: `data` (DICOM/lateralidade/série/
slice/labels/Dataset), `modeling` (backbone/head/modelo/ensemble),
`training` (loop de treino/validação), `inference` (TTA/geração de
submissão) e `cli` (entrypoints de linha de comando).
"""
