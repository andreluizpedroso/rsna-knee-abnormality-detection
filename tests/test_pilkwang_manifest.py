import json

import pytest

from src.rsna_knee.pilkwang.constants import SLOTS
from src.rsna_knee.pilkwang.manifest import (
    WeightsPackageError,
    load_manifest,
    validate_manifest,
)


def _pixel_group(**overrides):
    cfg = {
        "img": 336,
        "group": 3,
        "slices": 12,
        "crop_mm": 130.0,
        "band": [0.2, 0.8],
        "slots": [s[0] for s in SLOTS],
        "rules": {
            "order": "dominant_axis",
            "lat": "corner_x",
            "slot_fallback": True,
            "decode_fill": "zero",
        },
    }
    cfg.update(overrides)
    return json.dumps(cfg)


def test_load_manifest_validates_member_files(tmp_path):
    (tmp_path / "member.pt").write_bytes(b"x")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "members": [{
            "id": "abc",
            "file": "member.pt",
            "fold": 1,
            "pixel_group": _pixel_group(),
            "config": {"variant": "small", "pool": "cls_mean"},
            "holdout": 0.85,
        }]
    }))

    manifest = load_manifest(tmp_path)
    validate_manifest(manifest)

    assert len(manifest.members) == 1
    assert manifest.members[0].id == "abc"
    assert len(manifest.groups) == 1


def test_load_manifest_missing_member_file_raises(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "members": [{
            "id": "abc",
            "file": "missing.pt",
            "pixel_group": _pixel_group(),
            "config": {},
        }]
    }))

    with pytest.raises(WeightsPackageError):
        load_manifest(tmp_path)


def test_validate_manifest_rejects_unknown_slots(tmp_path):
    (tmp_path / "member.pt").write_bytes(b"x")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "members": [{
            "id": "abc",
            "file": "member.pt",
            "pixel_group": _pixel_group(slots=["WRONG"]),
            "config": {},
        }]
    }))

    manifest = load_manifest(tmp_path)
    with pytest.raises(WeightsPackageError):
        validate_manifest(manifest)

