"""Tests for ephys runtime models and loading service."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.io.datapackage_loader import ChannelTablePaths, ProbeInfo
from ephys_alignment_gui.services.ephys_data import (
    ChannelTable,
    EphysDataService,
    EphysStreamData,
)


def _probe_info(ephys_dir: Path, channel_table: ChannelTablePaths) -> ProbeInfo:
    return ProbeInfo(
        probe_id="rec1:streamA",
        probe_name="probeA",
        recording_id="rec1",
        logical_probe="probeA",
        ephys_collection="streamA",
        num_shanks=2,
        ephys_dir=ephys_dir,
        channel_table=channel_table,
        xyz_picks=(),
    )


def _write_channel_table_files(
    ephys_dir: Path,
    *,
    local_coordinates: np.ndarray,
    raw_ind: np.ndarray | None = None,
    contact_ids: np.ndarray | None = None,
    shank_indices: np.ndarray | None = None,
) -> ChannelTablePaths:
    ephys_dir.mkdir()
    local_path = ephys_dir / "channels.localCoordinates.npy"
    raw_path = ephys_dir / "channels.rawInd.npy"
    contact_path = ephys_dir / "channels.contactId.npy"
    shank_path = ephys_dir / "channels.shankInd.npy"
    np.save(local_path, local_coordinates)
    np.save(
        raw_path, raw_ind if raw_ind is not None else np.arange(len(local_coordinates))
    )
    if contact_ids is not None:
        np.save(contact_path, contact_ids)
    if shank_indices is not None:
        np.save(shank_path, shank_indices)
    return ChannelTablePaths(
        local_coordinates=local_path,
        raw_ind=raw_path,
        contact_id=contact_path if contact_ids is not None else None,
        shank_ind=shank_path,
    )


def test_channel_collection_is_row_view_into_stream():
    table = ChannelTable(
        local_coordinates=np.array(
            [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
        ),
        raw_ind=np.array([10, 11, 12, 13]),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )
    stream = EphysStreamData(
        recording_id="rec1",
        ephys_collection="streamA",
        ephys_dir=Path("/tmp/ephys"),
        channel_table=table,
        alf_data={"channels": {"exists": True}},
        session_notes="notes",
    )

    collection = stream.channel_collection(1)

    assert collection.stream is stream
    assert collection.rows.tolist() == [2, 3]
    assert collection.local_coordinates.tolist() == [[250.0, 0.0], [250.0, 20.0]]
    assert collection.depths.tolist() == [0.0, 20.0]
    assert collection.raw_ind.tolist() == [12, 13]
    assert collection.contact_ids.tolist() == ["s1e0", "s1e1"]


def test_channel_table_rejects_invalid_shank_index():
    table = ChannelTable(
        local_coordinates=np.array([[0.0, 0.0], [0.0, 20.0]]),
        shank_indices=np.array([0, 0]),
    )

    with pytest.raises(IndexError):
        table.rows_for_shank(1)


def test_channel_table_normalizes_absolute_shank_ids_to_local_indices():
    table = ChannelTable(
        local_coordinates=np.array([[250.0, 0.0], [250.0, 20.0]]),
        shank_indices=np.array([1, 1]),
    )

    assert table.n_shanks == 1
    assert table.shank_indices.tolist() == [0, 0]
    assert table.rows_for_shank(0).tolist() == [0, 1]


def test_ephys_data_service_loads_channel_table_from_probe_paths(tmp_path):
    local_coordinates = np.array([[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]])
    channel_paths = _write_channel_table_files(
        tmp_path / "ephys",
        local_coordinates=local_coordinates,
        raw_ind=np.array([0, 1, 2, 3]),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )

    table = EphysDataService().load_channel_table(
        _probe_info(tmp_path / "ephys", channel_paths)
    )

    assert table.n_shanks == 2
    assert table.rows_for_shank(1).tolist() == [2, 3]
    assert table.contact_ids.tolist() == ["s0e0", "s0e1", "s1e0", "s1e1"]


def test_ephys_data_service_loads_stream_data(monkeypatch, tmp_path):
    local_coordinates = np.array([[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]])
    channel_paths = _write_channel_table_files(
        tmp_path / "ephys",
        local_coordinates=local_coordinates,
        raw_ind=np.array([10, 11, 12, 13]),
        contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        shank_indices=np.array([0, 0, 1, 1]),
    )
    np.save(tmp_path / "ephys" / "spike_shank_indices.npy", np.array([0, 1, 1]))
    (tmp_path / "ephys" / "session_notes.txt").write_text("session notes")

    def fake_load_object(_folder, _object_name):
        return {}

    monkeypatch.setattr(
        "ephys_alignment_gui.services.ephys_data.alfio.load_object",
        fake_load_object,
    )

    stream = EphysDataService().load_stream_data(
        _probe_info(tmp_path / "ephys", channel_paths)
    )

    assert stream.stream_key == ("rec1", "streamA")
    assert stream.session_notes == "session notes"
    assert stream.alf_data["channels"]["localCoordinates"] is (
        stream.channel_table.local_coordinates
    )
    assert stream.alf_data["channels"]["rawInd"].tolist() == [10, 11, 12, 13]
    assert stream.alf_data["channels"]["contactId"].tolist() == [
        "s0e0",
        "s0e1",
        "s1e0",
        "s1e1",
    ]
    assert stream.alf_data["channels"]["shankInd"].tolist() == [0, 0, 1, 1]
    assert stream.alf_data["rms_AP"]["xaxis"] == "Time (s)"
    assert stream.alf_data["spike_shanks"].tolist() == [0, 1, 1]
    assert stream.channel_collection(1).depths.tolist() == [0.0, 20.0]


@pytest.fixture
def density_stream_loader(tmp_path, monkeypatch):
    ephys_dir = tmp_path / "ephys"
    channel_paths = _write_channel_table_files(
        ephys_dir, local_coordinates=np.array([[0.0, 0.0], [0.0, 20.0]])
    )
    service = EphysDataService()
    monkeypatch.setattr(service, "_load_alf_objects", lambda _: {})
    probe = _probe_info(ephys_dir, channel_paths)
    (ephys_dir / "band_corr").mkdir()
    (ephys_dir / "band_corr" / "row_channels.json").write_text(
        json.dumps(
            {"shanks": {"0": {"blocks": [{"label": "main", "rows": [0, 1]}]}}}
        )
    )
    return probe, service.load_stream_data


@pytest.mark.parametrize("advertised", [False, True])
def test_loads_raw_unit_density_without_pickle(density_stream_loader, monkeypatch, advertised):
    probe, load = density_stream_loader
    fallback = probe.ephys_dir / "_iblqc_ephysRawUnitDensity.density.npy"
    path = probe.ephys_dir / "manifest-density.npy" if advertised else fallback
    density = np.arange(12).reshape(2, 6)
    np.save(path, density)
    if advertised:
        np.save(fallback, np.zeros((1, 3)))
        probe = replace(probe, raw_unit_density=path)
    original_load = np.load
    calls = []

    def checked_load(file, *args, **kwargs):
        if file == path:
            calls.append(kwargs.get("allow_pickle"))
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", checked_load)
    entry = load(probe).alf_data["raw_unit_density"]

    assert entry["exists"] is True
    assert entry["path"] == path
    np.testing.assert_array_equal(entry["density"], density)
    assert calls == [False]


def test_absent_optional_density_is_quiet(density_stream_loader, caplog):
    probe, load = density_stream_loader
    entry = load(probe).alf_data["raw_unit_density"]
    assert entry == {
        "exists": False,
        "path": probe.ephys_dir / "_iblqc_ephysRawUnitDensity.density.npy",
    }
    assert not caplog.records


@pytest.mark.parametrize("unresolved", [False, True])
def test_failed_advertised_density_does_not_fall_back(
    density_stream_loader, caplog, unresolved
):
    probe, load = density_stream_loader
    np.save(
        probe.ephys_dir / "_iblqc_ephysRawUnitDensity.density.npy",
        np.ones((2, 4)),
    )
    path = probe.ephys_dir / "missing_density.npy"
    probe = replace(
        probe,
        raw_unit_density=path,
        raw_unit_density_error="Asset missing for missing_density.npy" if unresolved else None,
    )

    entry = load(probe).alf_data["raw_unit_density"]
    assert entry["exists"] is False
    assert entry["path"] == path
    assert "density" not in entry
    assert "missing_density.npy" in entry["error"]
    warnings = [record for record in caplog.records if "raw_unit_density" in record.message]
    assert len(warnings) == (0 if unresolved else 1)


@pytest.mark.parametrize(
    "density",
    [
        np.ones(4),
        np.empty((0, 4)),
        np.empty((2, 0)),
        np.ones((1, 2, 3)),
        np.array([[1j]]),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[object()]], dtype=object),
        np.array([["not numeric"]]),
        np.array([[True]]),
    ],
    ids=["1d", "empty-shanks", "empty-bins", "3d", "complex", "nan", "inf", "object", "text", "bool"],
)
def test_invalid_optional_density_does_not_block_stream(
    density_stream_loader, caplog, density
):
    probe, load = density_stream_loader
    path = probe.ephys_dir / "_iblqc_ephysRawUnitDensity.density.npy"
    np.save(path, density)
    advertised = probe.ephys_dir / "advertised-density.npy"
    path.rename(advertised)
    np.save(path, np.ones((2, 4)))
    path = advertised
    probe = replace(probe, raw_unit_density=path)

    stream = load(probe)
    entry = stream.alf_data["raw_unit_density"]
    assert stream.alf_data["channels"]["exists"]
    assert entry["exists"] is False
    assert "density" not in entry
    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert str(path) in message
    assert str(density.shape) in message
    assert str(density.dtype) in message
    assert "nonempty real numeric finite (n_shanks, n_bins)" in message


@pytest.mark.parametrize(
    "contents", [b"not a numpy file", b"\x93NUMPY", b"PK\x03\x04corrupt zip"]
)
def test_corrupt_optional_density_does_not_block_stream(
    density_stream_loader, caplog, contents
):
    probe, load = density_stream_loader
    path = probe.ephys_dir / "corrupt.npy"
    path.write_bytes(contents)
    probe = replace(probe, raw_unit_density=path)

    entry = load(probe).alf_data["raw_unit_density"]
    assert entry["exists"] is False
    assert "density" not in entry
    assert len(caplog.records) == 1
    assert str(path) in caplog.text
    assert "dims/dtype unavailable" in caplog.text
    assert "nonempty real numeric finite (n_shanks, n_bins)" in caplog.text


def test_optional_density_rejects_archive_instead_of_array(density_stream_loader, caplog):
    probe, load = density_stream_loader
    path = probe.ephys_dir / "archive.npy"
    with path.open("wb") as file:
        np.savez(file, density=np.ones((2, 4)))
    probe = replace(probe, raw_unit_density=path)

    assert load(probe).alf_data["raw_unit_density"]["exists"] is False
    assert len(caplog.records) == 1
    assert "dims=None, dtype=None" in caplog.text


def test_density_attaches_producer_row_channel_provenance(density_stream_loader):
    probe, load = density_stream_loader
    path = probe.ephys_dir / "_iblqc_ephysRawUnitDensity.density.npy"
    np.save(path, np.ones((1, 2)))
    provenance_path = probe.ephys_dir / "band_corr" / "row_channels.json"
    provenance = {
        "version": 3,
        "shanks": {
            "0": {
                "blocks": [
                    {"label": "main", "rows": [1]},
                    {"label": "surface", "rows": [0]},
                ]
            }
        },
    }
    provenance_path.write_text(json.dumps(provenance))

    entry = load(probe).alf_data["raw_unit_density"]
    assert entry["exists"] is True
    assert entry["row_channels_path"] == provenance_path
    assert entry["row_channels"] == provenance
    assert "provenance_error" not in entry


@pytest.mark.parametrize("contents", [None, "not json", "[]", "null", '{"shanks":'])
def test_missing_or_malformed_density_provenance_keeps_other_stream_data(
    density_stream_loader, caplog, contents
):
    probe, load = density_stream_loader
    path = probe.ephys_dir / "_iblqc_ephysRawUnitDensity.density.npy"
    np.save(path, np.ones((1, 2)))
    provenance_path = probe.ephys_dir / "band_corr" / "row_channels.json"
    if contents is None:
        provenance_path.unlink()
    else:
        provenance_path.write_text(contents)

    stream = load(probe)
    entry = stream.alf_data["raw_unit_density"]
    assert stream.alf_data["channels"]["exists"]
    assert entry["exists"] is True
    np.testing.assert_array_equal(entry["density"], np.ones((1, 2)))
    assert "row_channels" not in entry
    assert entry["row_channels_path"] == provenance_path
    assert str(provenance_path) in entry["provenance_error"]
    assert not caplog.records
