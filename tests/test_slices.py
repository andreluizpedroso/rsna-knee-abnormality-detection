"""Testes de src/rsna_knee/data/slices.py."""

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset

from src.rsna_knee.data.slices import order_slices_by_geometry, pick_middle_slice


def _touch_dcm_files(dir_path, names):
    for name in names:
        (dir_path / name).write_bytes(b"")


def _write_minimal_dicom(path, ipp):
    """DICOM sintético mínimo, só com o necessário pra `_slice_position`
    conseguir ler (ImagePositionPatient/ImageOrientationPatient) -- sem
    pixel_array, não precisamos de imagem real pra testar ordenação."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.ImagePositionPatient = ipp
    # Plano axial simples: linha ao longo de X, coluna ao longo de Y --
    # normal = X x Y = Z, então a posição ao longo de Z decide a ordem.
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.save_as(str(path), enforce_file_format=True)


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


def test_order_slices_by_geometry_sorts_by_real_position_not_filename(tmp_path):
    # Nomeados deliberadamente fora de ordem espacial: "a" (nome
    # alfabeticamente primeiro) fica no topo (Z=20), não embaixo.
    _write_minimal_dicom(tmp_path / "a_top.dcm", [0.0, 0.0, 20.0])
    _write_minimal_dicom(tmp_path / "b_bottom.dcm", [0.0, 0.0, 0.0])
    _write_minimal_dicom(tmp_path / "c_middle.dcm", [0.0, 0.0, 10.0])

    result = order_slices_by_geometry(tmp_path)

    assert result == [
        tmp_path / "b_bottom.dcm",
        tmp_path / "c_middle.dcm",
        tmp_path / "a_top.dcm",
    ]


def test_order_slices_by_geometry_falls_back_to_alphabetical_when_invalid(tmp_path):
    # Arquivos vazios/não-DICOM: geometria indisponível -> cai pro
    # fallback antigo (ordem alfabética de nome), sem lançar exceção.
    _touch_dcm_files(tmp_path, ["z.dcm", "a.dcm", "m.dcm"])
    result = order_slices_by_geometry(tmp_path)
    assert result == [tmp_path / "a.dcm", tmp_path / "m.dcm", tmp_path / "z.dcm"]


def test_order_slices_by_geometry_falls_back_when_only_one_slice_lacks_geometry(tmp_path):
    # Maioria com geometria valida, 1 sem -- ainda assim aborta a serie
    # inteira pro fallback (nao mistura ordens parcialmente confiaveis).
    _write_minimal_dicom(tmp_path / "b.dcm", [0.0, 0.0, 5.0])
    _write_minimal_dicom(tmp_path / "c.dcm", [0.0, 0.0, 10.0])
    _touch_dcm_files(tmp_path, ["a.dcm"])
    result = order_slices_by_geometry(tmp_path)
    assert result == [tmp_path / "a.dcm", tmp_path / "b.dcm", tmp_path / "c.dcm"]


def test_order_slices_by_geometry_empty_dir_returns_empty_list(tmp_path):
    assert order_slices_by_geometry(tmp_path) == []
