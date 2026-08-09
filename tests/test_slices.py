"""Testes de src/rsna_knee/data/slices.py -- só usa nomes de arquivo (não
precisa de conteúdo DICOM válido)."""

from src.rsna_knee.data.slices import pick_middle_slice


def _touch_dcm_files(dir_path, names):
    for name in names:
        (dir_path / name).write_bytes(b"")


def test_pick_middle_slice_odd_count(tmp_path):
    _touch_dcm_files(tmp_path, ["a.dcm", "b.dcm", "c.dcm"])
    result = pick_middle_slice(tmp_path)
    assert result == tmp_path / "b.dcm"


def test_pick_middle_slice_even_count(tmp_path):
    _touch_dcm_files(tmp_path, ["a.dcm", "b.dcm", "c.dcm", "d.dcm"])
    result = pick_middle_slice(tmp_path)
    assert result == tmp_path / "c.dcm"  # index len//2 = 2


def test_pick_middle_slice_empty_dir_returns_none(tmp_path):
    assert pick_middle_slice(tmp_path) is None


def test_pick_middle_slice_ignores_non_dcm_files(tmp_path):
    (tmp_path / "readme.txt").write_bytes(b"")
    assert pick_middle_slice(tmp_path) is None
