import numpy as np
import pytest

from src.rsna_knee import config
from src.rsna_knee.pilkwang.tta import aggregate_windows, window_starts


def test_window_starts_overlap_matches_pilkwang_contract():
    assert window_starts(12, group=3, overlap=True) == list(range(10))


def test_window_starts_non_overlap():
    assert window_starts(12, group=3, overlap=False) == [0, 3, 6, 9]


def test_aggregate_windows_uses_max_for_patch_targets_and_mean_for_others():
    n_targets = len(config.TARGET_COLUMNS)
    arr = np.zeros((3, n_targets), dtype=np.float32)
    fracture = config.TARGET_COLUMNS.index("Fracture")
    lateral_meniscus = config.TARGET_COLUMNS.index("Lateral Meniscus")
    acl = config.TARGET_COLUMNS.index("ACL")

    arr[:, fracture] = [0.1, 0.9, 0.2]
    arr[:, lateral_meniscus] = [0.4, 0.3, 0.8]
    arr[:, acl] = [0.1, 0.2, 0.9]

    out = aggregate_windows(arr)

    assert out[fracture] == pytest.approx(0.9)
    assert out[lateral_meniscus] == pytest.approx(0.8)
    assert out[acl] == pytest.approx(0.4)


def test_aggregate_windows_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_windows(np.zeros((0, len(config.TARGET_COLUMNS)), dtype=np.float32))

