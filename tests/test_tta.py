"""Testes de src/rsna_knee/inference/tta.py -- DICOM sintético mínimo (só
header, pra tta_slice_paths) e model/leitura de slice mockados (pra
predict_study_with_tta, sem precisar decodificar pixel_array real)."""

from unittest.mock import patch

import numpy as np
import pydicom
import torch
import torch.nn as nn
from pydicom.dataset import FileDataset, FileMetaDataset

from src.rsna_knee import config
from src.rsna_knee.inference.tta import predict_study_with_tta, tta_slice_paths


def _write_minimal_dicom(path, ipp):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.ImagePositionPatient = ipp
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.save_as(str(path), enforce_file_format=True)


def test_tta_slice_paths_picks_odd_window_centered_on_middle(tmp_path):
    for i, z in enumerate([0.0, 5.0, 10.0, 15.0, 20.0]):
        _write_minimal_dicom(tmp_path / f"s{i}.dcm", [0.0, 0.0, z])

    result = tta_slice_paths(tmp_path, n_windows=3)
    assert [p.name for p in result] == ["s1.dcm", "s2.dcm", "s3.dcm"]


def test_tta_slice_paths_returns_all_when_series_smaller_than_window(tmp_path):
    for i, z in enumerate([0.0, 5.0]):
        _write_minimal_dicom(tmp_path / f"s{i}.dcm", [0.0, 0.0, z])

    result = tta_slice_paths(tmp_path, n_windows=5)
    assert [p.name for p in result] == ["s0.dcm", "s1.dcm"]


def test_tta_slice_paths_clips_window_at_series_start(tmp_path):
    # Meio da serie fica perto do inicio (indice 1 de 0..4) -- janela de 3
    # nao pode comecar em -1, tem que ser clipada pro inicio real.
    for i, z in enumerate([0.0, 5.0, 10.0, 15.0, 20.0]):
        _write_minimal_dicom(tmp_path / f"s{i}.dcm", [0.0, 0.0, z])

    result = tta_slice_paths(tmp_path, n_windows=7)
    assert [p.name for p in result] == [f"s{i}.dcm" for i in range(5)]


def test_tta_slice_paths_empty_dir_returns_empty_list(tmp_path):
    assert tta_slice_paths(tmp_path) == []


class _SumModel(nn.Module):
    """Modelo de teste sem pesos reais -- retorna a soma dos pixels de
    cada item do batch, repetida pros N targets, só pra verificar que a
    agregação de TTA (média de sigmoid entre janelas) está correta."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vals = x.sum(dim=(1, 2, 3))
        return vals.unsqueeze(1).repeat(1, len(config.TARGET_COLUMNS)).float()


def test_predict_study_with_tta_averages_sigmoid_across_windows(tmp_path):
    model = _SumModel()
    device = torch.device("cpu")

    fake_paths = [tmp_path / "a.dcm", tmp_path / "b.dcm"]
    imgs_by_path = {
        fake_paths[0]: np.zeros((2, 2), dtype=np.float32),
        fake_paths[1]: np.full((2, 2), 2.0, dtype=np.float32),
    }

    with patch(
        "src.rsna_knee.inference.tta.series_mod.pick_series_dir", return_value=tmp_path
    ), patch(
        "src.rsna_knee.inference.tta.series_mod.lookup_anatomical_plane", return_value=None
    ), patch(
        "src.rsna_knee.inference.tta.tta_slice_paths", return_value=fake_paths
    ), patch(
        "src.rsna_knee.inference.tta.load_dicom_slice",
        side_effect=lambda p, image_size, plane: imgs_by_path[p],
    ):
        result = predict_study_with_tta(
            model, study_dir=tmp_path, study_id=None, series_df=None,
            device=device, image_size=2, n_windows=2,
        )

    # janela "a": soma = 0 (3 canais * 2x2 de zeros) -> sigmoid(0) = 0.5
    # janela "b": soma = 3*2*2*2 = 24 -> sigmoid(24) ~= 1.0
    expected = torch.sigmoid(torch.tensor([0.0, 24.0])).mean().item()
    assert result.shape == (len(config.TARGET_COLUMNS),)
    assert np.allclose(result, expected, atol=1e-5)


def test_predict_study_with_tta_raises_when_no_series_found(tmp_path):
    with patch(
        "src.rsna_knee.inference.tta.series_mod.pick_series_dir", return_value=None
    ):
        try:
            predict_study_with_tta(
                _SumModel(), study_dir=tmp_path, study_id=None, series_df=None,
                device=torch.device("cpu"),
            )
            assert False, "deveria ter levantado FileNotFoundError"
        except FileNotFoundError:
            pass
