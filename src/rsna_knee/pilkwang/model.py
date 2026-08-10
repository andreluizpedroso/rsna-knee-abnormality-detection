"""Arquitetura compatível com os members publicados pelo Pilkwang."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import constants
from .manifest import WeightsPackageError


class SlotHead(nn.Module):
    """Atenção por diagnóstico sobre os slots/séries de um estudo."""

    def __init__(
        self,
        dim: int,
        n_slot: int = constants.N_SLOT,
        n_out: int = len(constants.TARGETS),
        hidden: int = 256,
        p: float = 0.2,
        prior: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        self.prior = prior

        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == constants.N_SLOT and n_out == len(constants.TARGETS):
            for target, slots in constants.SLOT_PRIOR_TABLE.items():
                p_[constants.TARGETS.index(target), list(slots)] = constants.SLOT_PRIOR_STRENGTH
        if prior:
            self.register_buffer("slot_prior", p_)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden**0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class PilkwangModel(nn.Module):
    """DINOv2 encoder + `SlotHead`.

    Entrada: `imgs` com shape `(batch, n_slot, group, H, W)` em uint8 ou
    float no intervalo 0..255; `group` normalmente é 3 slices usados como
    canais do encoder.
    """

    def __init__(
        self,
        backbone: nn.Module,
        dim: int,
        pool: str = "cls_mean",
        prior: bool = False,
    ) -> None:
        super().__init__()
        if pool not in constants.POOL_PARTS:
            raise ValueError(f"pool desconhecido: {pool}")
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * constants.POOL_PARTS[pool], prior=prior)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        imgs: torch.Tensor,
        mask: torch.Tensor,
        img_size: int | None = None,
    ) -> torch.Tensor:
        b, s = imgs.shape[:2]
        x = imgs.reshape(b * s, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(b, s, -1)
        return self.head(feat, mask)


def build_transformers_dinov2(
    source: Path,
    unfreeze_last: int = 0,
    variant: str = "small",
    pool: str = "cls_mean",
    prior: bool = False,
) -> PilkwangModel:
    """Constrói o modelo com `transformers.AutoModel`.

    `source` deve apontar para o Kaggle Model `metaresearch/dinov2/...`.
    """
    from transformers import AutoModel

    if source is None or not Path(source).exists():
        raise FileNotFoundError(f"DINOv2 {variant} não encontrado em {source}")
    backbone = AutoModel.from_pretrained(str(source))
    n_layer = len(backbone.encoder.layer)
    for prm in backbone.parameters():
        prm.requires_grad = False
    for block in backbone.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in block.parameters():
            prm.requires_grad = True
    for prm in backbone.layernorm.parameters():
        prm.requires_grad = True
    return PilkwangModel(
        backbone,
        dim=backbone.config.hidden_size,
        pool=pool,
        prior=prior,
    )


@torch.no_grad()
def fingerprint(
    model: nn.Module,
    device: torch.device,
    img_size: int,
    n_slot: int = constants.N_SLOT,
    group: int = constants.GROUP,
    seed: int = 2026,
) -> np.ndarray:
    """Saída do modelo em um saco sintético fixo, usada para checar pesos."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    imgs = torch.randint(
        0,
        256,
        (1, n_slot, group, constants.CACHE_IMG, constants.CACHE_IMG),
        dtype=torch.uint8,
        generator=gen,
    )
    mask = torch.ones(1, n_slot)
    model.eval()
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        out = model(imgs.to(device), mask.to(device), img_size).float()
    return out.detach().cpu().numpy().reshape(-1)


def check_fingerprint(
    model: nn.Module,
    device: torch.device,
    img_size: int,
    expected: list[float] | np.ndarray,
    tol: float = 2e-3,
    tag: str = "",
) -> float:
    """Compara o fingerprint calculado contra o armazenado no checkpoint."""
    got = fingerprint(model, device, img_size)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise WeightsPackageError(
            f"{tag}fingerprint shape {got.shape} != armazenado {exp.shape}"
        )
    diff = float(np.abs(got - exp).max())
    if diff > tol:
        raise WeightsPackageError(
            f"{tag}fingerprint difere por {diff:.4g} (tol {tol:g}); "
            "arquitetura/preprocessamento não batem com os pesos"
        )
    return diff
