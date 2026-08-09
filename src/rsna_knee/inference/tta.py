"""Test-time augmentation por janelas de slice sobrepostas.

Ainda não implementado -- hoje a inferência usa 1 slice fixo (o do meio,
ver `data/slices.pick_middle_slice`) por estudo. Roadmap (ver
CLAUDE.md/PROGRESS.md): rodar o forward em slices vizinhos ao central (já
ordenados geometricamente, ver roadmap de `data/slices.py`) e agregar as
probabilidades previstas, em vez de depender de um único slice; combinado
com ensemble por rank-mean entre checkpoints (troca a agregação atual de
`inference/submission.predict_ensemble`, que é média de sigmoid) pra
neutralizar diferenças de calibração entre modelos treinados
separadamente.
"""
