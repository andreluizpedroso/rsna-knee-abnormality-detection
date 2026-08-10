"""Inferência completa para pacote de pesos Pilkwang."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import config
from ..inference.submission import write_submission
from . import constants
from .manifest import WeightsManifest
from .model import build_transformers_dinov2, check_fingerprint
from .pixels import (
    PixelRuntime,
    annotate_headers,
    build_cache,
    laterality_by_study,
    pick_slots,
    walk_series,
)
from .tta import aggregate_windows, window_starts


@torch.no_grad()
def predict_member(
    model: torch.nn.Module,
    cache: np.ndarray,
    mask: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
    img_size: int,
    group: int,
    starts: list[int],
    eval_batch: int = constants.EVAL_BATCH,
) -> np.ndarray:
    """Prediz um member, agregando janelas TTA com patch por target."""
    if not starts:
        raise ValueError("predict_member precisa de pelo menos uma janela TTA")
    model.eval()
    out = []
    for b in range(0, len(idx), eval_batch):
        sel = idx[b:b + eval_batch]
        m = torch.from_numpy(mask[sel]).to(device)
        per_window = []
        for start in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(cache[sel, :, start:start + group])
            ).to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(rows, m, img_size).float()
            per_window.append(torch.sigmoid(logits).cpu().numpy())
        stacked = np.stack(per_window, axis=0)  # windows, batch, targets
        batch = np.stack([aggregate_windows(stacked[:, i, :]) for i in range(stacked.shape[1])])
        out.append(batch)
    return np.concatenate(out, axis=0) if out else np.zeros((0, len(constants.TARGETS)), np.float32)


def infer_from_manifest(
    manifest: WeightsManifest,
    data_root: Path,
    dinov2_root: Path,
    out_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    """Roda o ensemble publicado e escreve `submission.csv`."""
    started = time.time()
    test_df = pd.read_csv(Path(data_root) / "test.csv")
    test_series = pd.read_csv(Path(data_root) / "test_series.csv")
    plane_map = dict(zip(test_series["SeriesInstanceUID"], test_series["Anatomical_Plane"]))

    headers = annotate_headers(walk_series(data_root, "test"))
    per_member: list[dict[str, object]] = []

    for gi, (pixel_group, members) in enumerate(manifest.groups.items(), 1):
        pixel_config = json.loads(pixel_group)
        runtime = PixelRuntime(pixel_config)
        print(
            f"decode group {gi}/{len(manifest.groups)}: {runtime.img}px, "
            f"{runtime.cache_slices} slices, {len(members)} member(s)"
        )
        slots = pick_slots(headers, plane_map, runtime)
        lat = laterality_by_study(headers, runtime)
        study_ids, cache, mask = build_cache(slots, plane_map, lat, runtime)
        idx = np.arange(len(study_ids))
        starts = window_starts(cache.shape[2], runtime.group)

        for member in sorted(members, key=lambda m: -(m.holdout or 0.0)):
            t0 = time.time()
            ckpt = torch.load(manifest.root / member.file, map_location="cpu", weights_only=False)
            model = build_transformers_dinov2(
                source=dinov2_root,
                unfreeze_last=int(member.config.get("unfreeze_last", 0)),
                variant=str(member.config.get("variant", "small")),
                pool=str(member.config.get("pool", "cls_mean")),
                prior=bool(member.config.get("prior", False)),
            ).to(device)
            model.load_state_dict(ckpt["model"])
            check_fingerprint(model, device, runtime.img, ckpt["fingerprint"], tag=f"{member.id}: ")
            pred = predict_member(
                model,
                cache,
                mask,
                idx,
                device,
                runtime.img,
                runtime.group,
                starts,
            )
            per_member.append({"id": member.id, "ids": study_ids, "pred": pred})
            print(
                f"  {member.id} fold {member.fold}: {len(study_ids)} studies, "
                f"{len(starts)} window(s), {time.time() - t0:.0f}s"
            )
            del model, ckpt
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del cache, mask
        gc.collect()

    all_ids = sorted({sid for item in per_member for sid in item["ids"]})
    pos = {sid: i for i, sid in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(constants.TARGETS)), np.float64)
    for item in per_member:
        ranks = pd.DataFrame(item["pred"]).rank(pct=True).to_numpy()
        acc[[pos[sid] for sid in item["ids"]]] += ranks
    acc /= max(len(per_member), 1)

    sub = write_submission(all_ids, acc, out_path)
    sub = test_df[[config.ID_COLUMN]].merge(sub, on=config.ID_COLUMN, how="left")
    sub[config.TARGET_COLUMNS] = sub[config.TARGET_COLUMNS].fillna(0.5)
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    sub.to_csv(out_path, index=False)
    print(
        f"submission.csv = rank mean of {len(per_member)} member(s); "
        f"{sub.shape}; nulls {int(sub[config.TARGET_COLUMNS].isna().sum().sum())}; "
        f"{time.time() - started:.0f}s"
    )
    return sub

