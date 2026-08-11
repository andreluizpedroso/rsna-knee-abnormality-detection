# RSNA Knee Abnormality Detection

Trabalho na competição Kaggle
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
detectar 12 anormalidades clinicamente relevantes no joelho a partir de
MRI.

## O desafio

- **Entrada**: séries de ressonância magnética (DICOM) do joelho, por
  estudo, com múltiplas séries/planos anatômicos por estudo.
- **Saída**: probabilidade (0–1) de 12 anormalidades por estudo — ACL, MCL,
  Meniscus (medial/lateral), OA (medial/lateral/patellofemoral), Effusion,
  Synovitis, Baker's cyst, Contusion, Fracture.
- **Métrica**: AUC-ROC macro entre os 12 targets.
- **Formato**: *code competition* — a submissão final roda dentro de um
  Kaggle Notebook, sem acesso à internet durante a execução, runtime ≤ 9h.
- **Prêmio**: existe uma trilha adicional de eficiência que pondera AUC vs.
  tempo de execução.

## Dados

- `train.csv`: uma linha por estudo (`StudyInstanceUID`, `Report`, 12
  colunas de label). Só uma pequena fração dos estudos de treino tem as 12
  colunas de label preenchidas — o restante vem sem label direto.
  `PatientSex` não é mais uma coluna do CSV (removida pelo host — já vem no
  header DICOM).
- `train_series.csv`: uma linha por série (`SeriesInstanceUID`,
  `Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression`).
- `Report`: texto livre do laudo radiológico, multi-idioma, presente em
  100% dos estudos de treino. **Não existe na base de teste** (confirmado
  pelo host).
- O gabarito (labels "gold") vem sempre de leitura de imagem por
  radiologistas MSK, nunca do texto do laudo (confirmado pelo host).
- Dataset completo: 819.640 arquivos DICOM, 569.76 GB.

## Baixar dados

Requer aceitar as regras da competição no Kaggle e ter `kaggle.json`
configurado:

```bash
bash scripts/download_data.sh          # CSVs (metadados/labels/reports)
bash scripts/download_data.sh --full   # dataset completo (569.76 GB)
python scripts/download_sample_images.py --n 5   # amostra mínima
```
