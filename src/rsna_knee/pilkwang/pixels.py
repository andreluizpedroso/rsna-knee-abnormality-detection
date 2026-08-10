"""Pré-processamento de pixels compatível com a referência Pilkwang v15."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

from . import constants

HDR_TAGS = [
    "SeriesDescription",
    "SequenceName",
    "ScanOptions",
    "ScanningSequence",
    "RepetitionTime",
    "EchoTime",
    "Laterality",
    "ImageLaterality",
    "PixelSpacing",
    "Rows",
    "Columns",
    "RescaleSlope",
    "RescaleIntercept",
    "ImagePositionPatient",
    "ImageOrientationPatient",
]

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
    r"water excit|\btirm\b|\bsting\b|\bfatsup\b"
)
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")

LEGACY_LAT_OFFSET_MM = 5.0
LAT_MIN_OFFSET_MM = 20.0
ORDER_TAGS = [(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0013)]


class PixelRuntime:
    """Config efetiva do grupo de pixels que será decodificado."""

    def __init__(self, pixel_config: dict[str, Any]) -> None:
        self.img = int(pixel_config["img"])
        self.group = int(pixel_config["group"])
        self.cache_slices = int(pixel_config["slices"])
        self.crop_mm = float(pixel_config["crop_mm"])
        self.slice_band = tuple(float(x) for x in pixel_config["band"])
        self.rules = {**constants.RULES_NATIVE, **(pixel_config.get("rules") or {})}
        self.slots = list(constants.SLOTS)

    @property
    def n_group(self) -> int:
        return max(self.cache_slices // self.group, 1)


def _hdr_vec(value: Any, n: int) -> np.ndarray | None:
    if not isinstance(value, str):
        return None
    try:
        values = [float(x) for x in value.split("|")]
    except ValueError:
        return None
    return np.array(values) if len(values) >= n else None


def probe(item: tuple[str, str, str, str]) -> dict[str, Any]:
    """Lê um header representativo de uma série."""
    split, study, series, path = item
    row: dict[str, Any] = {
        "split": split,
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "dir": path,
    }
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(
            os.path.join(path, files[len(files) // 2]),
            stop_before_pixels=True,
            force=True,
        )
        for tag in HDR_TAGS:
            value = getattr(ds, tag, None)
            if value is None:
                row[tag] = None
            elif isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
                row[tag] = "|".join(str(x) for x in value)
            else:
                row[tag] = str(value)
    except Exception as exc:
        row["error"] = repr(exc)
        row.setdefault("files", [])
        row.setdefault("n_slices", 0)
    return row


def walk_series(root: Path, split: str, max_workers: int = 16) -> pd.DataFrame:
    """Lista e sonda todas as séries de `root/<split>_series`."""
    base = Path(root) / f"{split}_series"
    items = []
    for study in os.scandir(base):
        if not study.is_dir():
            continue
        for series in os.scandir(study.path):
            if series.is_dir():
                items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Recupera fat-sat e weighting a partir de headers DICOM."""
    out = df.copy()
    desc = (out["SeriesDescription"].fillna("") + " " + out["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)
    opts = out["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))

    out["fatsat"] = opts_fs | desc.str.contains(_FATSAT_RX, regex=True)
    out["weight"] = "UNK"
    out.loc[desc.str.contains(_T1_RX, regex=True), "weight"] = "T1"
    out.loc[desc.str.contains(_T2_RX, regex=True), "weight"] = "T2"
    out.loc[desc.str.contains(_PD_RX, regex=True), "weight"] = "PD"

    tr = pd.to_numeric(out["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(out["EchoTime"], errors="coerce")
    out.loc[(out["weight"] == "UNK") & (tr < 800), "weight"] = "T1"
    out.loc[(out["weight"] == "UNK") & (tr >= 800) & (te >= 40), "weight"] = "T2"
    out.loc[(out["weight"] == "UNK") & (tr >= 800) & (te < 40), "weight"] = "PD"
    out["fluid"] = np.isin(out["weight"], ["PD", "T2"])
    out["px"] = pd.to_numeric(
        out["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce",
    )
    return out


def pick_slots(
    series_df: pd.DataFrame,
    plane_map: dict[str, str],
    runtime: PixelRuntime,
) -> dict[str, dict[str, pd.Series]]:
    """Escolhe uma série por slot por estudo."""
    df = series_df.copy()
    df["plane"] = df["SeriesInstanceUID"].map(plane_map)
    out: dict[str, dict[str, pd.Series]] = {}
    for study, group in df.groupby("StudyInstanceUID"):
        chosen: dict[str, pd.Series] = {}
        for name, plane, want_fatsat, want_fluid in runtime.slots:
            cand = group[group["plane"] == plane]
            if want_fluid is not None:
                cand = cand[cand["fluid"] == bool(want_fluid)]
            if want_fatsat is not None:
                cand = cand[cand["fatsat"] == bool(want_fatsat)]
            if len(cand) == 0 and runtime.rules["slot_fallback"] and want_fatsat is False:
                cand = group[(group["plane"] == plane) & (~group["fatsat"])]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out


def _natural_key(name: str) -> tuple[Any, ...]:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(name)))


def _order_dominant_axis(rec: pd.Series) -> tuple[list[str], bool]:
    files, directory = rec["files"], rec["dir"]
    rows = []
    for pos, file in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(
                os.path.join(directory, file),
                force=True,
                stop_before_pixels=True,
                specific_tags=["ImagePositionPatient", "InstanceNumber"],
            )
            raw = getattr(ds, "ImagePositionPatient", None)
            if raw is not None and len(raw) >= 3:
                coords = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(coords).all():
                    ipp = coords
            n = getattr(ds, "InstanceNumber", None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((file, ipp, inst, pos))

    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (
            float(r[1][axis]) if r[1] is not None else spare,
            r[2] if r[2] is not None else float("inf"),
            r[3],
        ))
    elif sum(r[2] is not None for r in rows) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float("inf"), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return [r[0] for r in rows], True


def order_slices(rec: pd.Series, runtime: PixelRuntime) -> tuple[list[str], bool]:
    """Ordena slices conforme as regras nativas ou legadas do member."""
    if runtime.rules["order"] == "dominant_axis":
        return _order_dominant_axis(rec)

    files, directory = rec["files"], rec["dir"]
    keyed = []
    for file in files:
        key = None
        try:
            ds = pydicom.dcmread(
                os.path.join(directory, file),
                force=True,
                stop_before_pixels=True,
                specific_tags=ORDER_TAGS,
            )
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            key = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                key = float(ds.InstanceNumber)
            except Exception:
                key = None
        keyed.append((key, file))
    if any(k is None for k, _ in keyed):
        return files, False
    return [f for _, f in sorted(keyed, key=lambda t: t[0])], True


def read_slot(rec: pd.Series, runtime: PixelRuntime) -> torch.Tensor | None:
    """Lê `runtime.cache_slices` de uma série para uint8 `[slice, H, W]`."""
    files = rec.get("ordered") or rec["files"]
    directory = rec["dir"]
    n = len(files)
    if n == 0:
        return None

    lo = int(runtime.slice_band[0] * (n - 1))
    hi = int(runtime.slice_band[1] * (n - 1))
    idx = np.unique(np.linspace(lo, hi, runtime.cache_slices).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < runtime.cache_slices:
        idx = np.append(idx, idx[-1])

    planes = []
    for i in idx[:runtime.cache_slices]:
        try:
            ds = pydicom.dcmread(os.path.join(directory, files[int(i)]), force=True)
            arr = ds.pixel_array.astype(np.float32)
            arr = arr * float(getattr(ds, "RescaleSlope", 1) or 1)
            arr = arr + float(getattr(ds, "RescaleIntercept", 0) or 0)
        except Exception:
            arr = None
        planes.append(arr)

    got = [k for k, plane in enumerate(planes) if plane is not None]
    if runtime.rules["decode_fill"] == "zero":
        planes = [
            np.zeros((runtime.img, runtime.img), np.float32) if plane is None else plane
            for plane in planes
        ]
        got = list(range(len(planes)))
    if not got:
        return None
    if len(got) < len(planes):
        for k, plane in enumerate(planes):
            if plane is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]

    shape = planes[0].shape
    planes = [p if p.shape == shape else np.zeros(shape, np.float32) for p in planes]
    vol = np.stack(planes)

    px = rec.get("px")
    if px and np.isfinite(px) and px > 0:
        want = int(round(runtime.crop_mm / px))
        h, w = shape
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(runtime.img, runtime.img),
        mode="bilinear",
        align_corners=False,
    )
    return (tensor.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


def _side_from_corner_x(headers: pd.DataFrame) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for study, group in headers.groupby("StudyInstanceUID"):
        xs = []
        for row in group.itertuples(index=False):
            ipp = _hdr_vec(getattr(row, "ImagePositionPatient", None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[study] = None
            continue
        x = float(np.median(xs))
        out[study] = None if abs(x) < LEGACY_LAT_OFFSET_MM else ("R" if x < 0 else "L")
    return out


def _side_from_geometry(headers: pd.DataFrame) -> dict[str, str | None]:
    cx: dict[str, list[float]] = {}
    for row in headers.itertuples(index=False):
        ipp = _hdr_vec(getattr(row, "ImagePositionPatient", None), 3)
        iop = _hdr_vec(getattr(row, "ImageOrientationPatient", None), 6)
        ps = _hdr_vec(getattr(row, "PixelSpacing", None), 2)
        rows = getattr(row, "Rows", None)
        cols = getattr(row, "Columns", None)
        if ipp is None or iop is None or ps is None or not rows or not cols:
            continue
        try:
            center = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(row.StudyInstanceUID, []).append(float(center[0]))
    out = {}
    for study, xs in cx.items():
        x = float(np.median(xs))
        out[study] = None if abs(x) < LAT_MIN_OFFSET_MM else ("R" if x < 0 else "L")
    return out


def laterality_by_study(headers: pd.DataFrame, runtime: PixelRuntime) -> dict[str, str | None]:
    """Study -> L/R/None, com regra nativa ou legada."""
    geo = _side_from_corner_x(headers) if runtime.rules["lat"] == "corner_x" else _side_from_geometry(headers)
    out: dict[str, str | None] = {}
    for study, group in headers.groupby("StudyInstanceUID"):
        values = [str(x).strip().upper() for x in group["Laterality"].dropna()]
        if runtime.rules["lat"] == "corner_x" and "ImageLaterality" in group.columns:
            values += [str(x).strip().upper() for x in group["ImageLaterality"].dropna()]
        values = [x[0] for x in values if x and x[0] in ("L", "R")]
        out[study] = values[0] if values else geo.get(study)
    return out


def normalise_laterality(img: torch.Tensor, plane: str, laterality: str | None) -> torch.Tensor:
    """Mapeia joelhos para a convenção de joelho esquerdo da referência."""
    if laterality != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


def build_cache(
    slot_map: dict[str, dict[str, pd.Series]],
    plane_map: dict[str, str],
    lat_map: dict[str, str | None],
    runtime: PixelRuntime,
    max_workers: int = 12,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Decodifica todos os slots escolhidos para cache uint8."""
    studies = sorted(slot_map)
    sidx = {study: i for i, study in enumerate(studies)}
    cache = np.zeros(
        (len(studies), constants.N_SLOT, runtime.cache_slices, runtime.img, runtime.img),
        np.uint8,
    )
    mask = np.zeros((len(studies), constants.N_SLOT), np.float32)

    jobs = [
        (study, k, plane, slot_map[study][name])
        for study in studies
        for k, (name, plane, _, _) in enumerate(runtime.slots)
        if name in slot_map[study]
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        ordered = list(pool.map(lambda job: order_slices(job[3], runtime), jobs))
    for (_, _, _, rec), (files, _) in zip(jobs, ordered):
        rec["ordered"] = files

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        decoded = list(pool.map(lambda job: read_slot(job[3], runtime), jobs))
    for (study, k, plane, _), img in zip(jobs, decoded):
        if img is None:
            continue
        cache[sidx[study], k] = normalise_laterality(img, plane, lat_map.get(study)).numpy()
        mask[sidx[study], k] = 1.0
    gc.collect()
    return studies, cache, mask


def cache_tag(pixel_config: dict[str, Any]) -> str:
    """Nome legível/estável para um grupo de pixels."""
    cfg = PixelRuntime(pixel_config)
    tag = (
        f"{cfg.img}px_{cfg.cache_slices}sl_{int(cfg.crop_mm)}mm_"
        f"{cfg.slice_band[0]:.2f}-{cfg.slice_band[1]:.2f}"
    )
    if cfg.rules != constants.RULES_NATIVE:
        tag += "_" + hashlib.md5(json.dumps(cfg.rules, sort_keys=True).encode()).hexdigest()[:6]
    return tag

