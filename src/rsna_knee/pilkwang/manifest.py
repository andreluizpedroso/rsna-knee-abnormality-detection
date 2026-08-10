"""Leitura e validação do pacote público de pesos Pilkwang."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import constants


class WeightsPackageError(RuntimeError):
    """Erro fatal quando o pacote de pesos não pode ser usado com segurança."""


@dataclass(frozen=True)
class WeightMember:
    """Um member do ensemble publicado."""

    id: str
    file: str
    fold: int | None
    pixel_group: str
    config: dict[str, Any]
    holdout: float | None = None
    annot: float | None = None

    @property
    def pixel_config(self) -> dict[str, Any]:
        try:
            return json.loads(self.pixel_group)
        except json.JSONDecodeError as exc:
            raise WeightsPackageError(
                f"member {self.id!r} tem pixel_group inválido"
            ) from exc


@dataclass(frozen=True)
class WeightsManifest:
    """Manifest completo do pacote de pesos."""

    root: Path
    members: list[WeightMember]

    @property
    def groups(self) -> dict[str, list[WeightMember]]:
        out: dict[str, list[WeightMember]] = {}
        for member in self.members:
            out.setdefault(member.pixel_group, []).append(member)
        return out


def _member_from_dict(raw: dict[str, Any]) -> WeightMember:
    missing = [k for k in ("id", "file", "pixel_group", "config") if k not in raw]
    if missing:
        raise WeightsPackageError(f"member sem campo(s) obrigatório(s): {missing}")
    return WeightMember(
        id=str(raw["id"]),
        file=str(raw["file"]),
        fold=int(raw["fold"]) if raw.get("fold") is not None else None,
        pixel_group=str(raw["pixel_group"]),
        config=dict(raw["config"]),
        holdout=float(raw["holdout"]) if raw.get("holdout") is not None else None,
        annot=float(raw["annot"]) if raw.get("annot") is not None else None,
    )


def load_manifest(root: Path) -> WeightsManifest:
    """Carrega `manifest.json` e garante que os checkpoints existem."""
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file():
        raise WeightsPackageError(f"manifest.json não encontrado em {root}")
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WeightsPackageError(f"não consegui ler {path}") from exc

    members_raw = raw.get("members")
    if not isinstance(members_raw, list) or not members_raw:
        raise WeightsPackageError("manifest.json não contém lista não-vazia de members")

    members = [_member_from_dict(m) for m in members_raw]
    missing_files = [m.file for m in members if not (root / m.file).is_file()]
    if missing_files:
        raise WeightsPackageError(
            f"{len(missing_files)} checkpoint(s) listados no manifest não existem; "
            f"primeiro ausente: {missing_files[0]}"
        )
    return WeightsManifest(root=root, members=members)


def find_weights_root(base: Path = Path("/kaggle/input")) -> Path | None:
    """Procura um pacote com `manifest.json` válido em um mount Kaggle."""
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "manifest.json" not in files:
            continue
        candidate = Path(root)
        try:
            load_manifest(candidate)
        except WeightsPackageError:
            continue
        return candidate
    return None


def validate_pixel_config(pixel_config: dict[str, Any]) -> None:
    """Recusa configs que este pacote ainda não sabe reproduzir."""
    slots = list(pixel_config.get("slots", []))
    expected_slots = [s[0] for s in constants.SLOTS]
    if slots != expected_slots:
        raise WeightsPackageError(
            f"slots do peso {slots} != slots suportados {expected_slots}"
        )

    pool_img = int(pixel_config.get("img", constants.CACHE_IMG))
    if pool_img not in (224, 336):
        raise WeightsPackageError(f"resolução de member não suportada: {pool_img}")

    rules = pixel_config.get("rules") or constants.RULES_NATIVE
    unknown = {
        k: v
        for k, v in rules.items()
        if k not in constants.RULES_NATIVE
        or v not in (constants.RULES_NATIVE[k], constants.RULES_LEGACY[k])
    }
    if unknown:
        raise WeightsPackageError(
            f"pixel rules não suportadas pelos módulos atuais: {unknown}"
        )


def validate_manifest(manifest: WeightsManifest) -> None:
    """Valida todos os grupos de pixel declarados pelo manifest."""
    for member in manifest.members:
        validate_pixel_config(member.pixel_config)

