"""Ensemble entre modelos/checkpoints (ex.: folds diferentes de um mesmo
treino, ou seeds diferentes).

Ainda não implementado -- hoje o ensemble usado em produção é a média
simples de sigmoids feita em `inference/submission.py` (`predict_ensemble`).
Roadmap (ver CLAUDE.md/PROGRESS.md): ensemble via múltiplos seeds/folds com
ponderação por holdout (persistir a métrica de validação junto de cada
checkpoint e ponderar a contribuição de cada um proporcionalmente a essa
métrica) entra aqui quando implementado.
"""
