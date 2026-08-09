"""Agregação por atenção (MIL -- multiple instance learning) sobre
embeddings de múltiplas instâncias (séries) de um mesmo estudo.

Roadmap item 7 (ver CLAUDE.md): tratar o estudo como um "saco" de
instâncias -- extrair embeddings de cada série com o MESMO backbone
(pesos compartilhados, não 1 backbone por instância) e agregar via uma
camada de atenção (pesos de importância aprendidos por instância,
normalizados via softmax, combinados por soma ponderada) em vez de pooling
fixo (média/max), permitindo que o modelo aprenda a dar mais peso a
séries mais informativas por label, sem supervisão direta de qual
instância importa.
"""

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Agrega `(batch, n_instances, feat_dim)` num único vetor por estudo
    `(batch, feat_dim)`, por soma ponderada com pesos aprendidos (attention
    MIL, Ilse et al. 2018).

    `mask` (`(batch, n_instances)`, 1 = instância válida, 0 = padding)
    permite lidar com "sacos" de tamanho variável (estudo com menos séries
    que o máximo do batch) sem mudar o shape do batch -- instâncias
    mascaradas recebem `-inf` de score antes do softmax, então saem com
    peso de atenção efetivamente 0."""

    def __init__(self, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.attention(embeddings).squeeze(-1)  # (batch, n_instances)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (batch, n_instances)
        return torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)  # (batch, feat_dim)
