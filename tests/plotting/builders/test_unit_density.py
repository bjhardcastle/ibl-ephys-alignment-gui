"""Raw unit-density coverage and physical-depth payload regressions."""

from pathlib import Path

import numpy as np
import pytest

from ephys_alignment_gui.plotting.payload_cache import EphysPlotPayloadCache
from ephys_alignment_gui.plotting.payload_cache_factory import (
    EphysPlotPayloadCacheFactory,
)
from ephys_alignment_gui.plotting.registry import (
    available_plot_specs_for_menu,
    default_plot_spec,
    resolve_plot_payload,
)
from ephys_alignment_gui.runtime.ephys_stream import EphysStreamRuntime
from ephys_alignment_gui.services.ephys_data import ChannelTable, EphysStreamData


def _cache(coords, density, shanks=None, active=0, runtime=True, provenance=True):
    table = ChannelTable(np.asarray(coords, dtype=float), shank_indices=shanks)
    data = {
        "channels": {"localCoordinates": table.local_coordinates},
        "spikes": {
            "exists": False,
            "clusters": np.array([], dtype=int),
            "depths": np.array([]),
            "amps": np.array([]),
        },
        "clusters": {
            "exists": False,
            "metrics": {"default_qc": np.array([], dtype=bool)},
        },
    }
    if table.shank_indices is not None:
        data["channels"]["shankInd"] = table.shank_indices
    if density is not None:
        if provenance is True:
            provenance = {
                "version": 3,
                "shanks": {
                    str(idx): {
                        "blocks": [
                            {
                                "label": "main",
                                "rows": table.rows_for_shank(idx).tolist(),
                            }
                        ]
                    }
                    for idx in range(table.n_shanks)
                },
            }
        data["raw_unit_density"] = {
            "exists": True,
            "density": np.asarray(density),
            "path": "/input/_iblqc_ephysRawUnitDensity.density.npy",
            "row_channels": provenance,
            "row_channels_path": "/input/band_corr/row_channels.json",
        }
    stream = EphysStreamData("rec", "stream", Path("/input"), table, data, "")
    return EphysPlotPayloadCache(
        stream.ephys_dir,
        data,
        active,
        channel_collection=stream.channel_collection(active) if runtime else None,
    )


def _paired(depths, x=0):
    return np.array([[lateral, depth] for depth in depths for lateral in (x, x + 32)])


def _bank_depths(banks=1, pitch=20):
    return np.arange(round(banks * 3840 / pitch)) * pitch


@pytest.mark.parametrize("runtime", [True, False])
def test_np1_paired_bins_use_sorted_physical_depths_tip_at_bottom(runtime):
    depths = _bank_depths()
    coords = _paired(depths)[::-1]
    density = np.linspace(0.1, 0.3, depths.size)
    cache = _cache(coords, [density], runtime=runtime)
    specs = available_plot_specs_for_menu(cache, "line")
    assert [spec.key for spec in specs] == ["line.unit_density"]
    assert specs[0].label == "Unit density"
    assert specs[0].renderer == "line"
    assert default_plot_spec("line").key == "line.fr"
    payload = resolve_plot_payload(cache, "line.unit_density")
    np.testing.assert_array_equal(payload["x"], density)
    # The ordinary line renderer has x=density, y=depth in µm, increasing up.
    np.testing.assert_array_equal(payload["y"], depths)
    assert payload["xaxis"] == "Unit density"
    assert payload["xrange"][0] == 0
    assert payload["xrange"][1] >= 0.3
    assert resolve_plot_payload(cache, "line.unit_density") is payload


def test_multiple_shanks_select_local_row_and_validate_whole_stream():
    depths = _bank_depths()
    coords = np.vstack([_paired(depths), _paired(depths, x=250)])
    shanks = [2] * 384 + [3] * 384
    density = np.vstack([np.arange(192), np.arange(192) + 1])
    for active in (0, 1):
        cache = _cache(coords, density, shanks, active)
        payload = resolve_plot_payload(cache, "line.unit_density")
        np.testing.assert_array_equal(payload["x"], density[active])
        np.testing.assert_array_equal(payload["y"], depths)
    # A different shank's coverage is invalid even when the active row matches.
    cache = _cache(np.vstack([coords, [[250, 3840]]]), density, shanks + [3])
    assert resolve_plot_payload(cache, "line.unit_density") is None


def test_split_stream_uses_stream_shanks_not_physical_probe_shank_ids():
    depths = _bank_depths()
    density = np.arange(192)
    cache = _cache(_paired(depths, x=750), [density], [3] * 384)
    payload = resolve_plot_payload(cache, "line.unit_density")
    np.testing.assert_array_equal(payload["x"], density)
    np.testing.assert_array_equal(payload["y"], depths)


def test_multibank_full_coverage_uses_all_physical_rows():
    depths = _bank_depths(2)
    coords = np.vstack([_paired(depths[:192]), _paired(depths[192:], x=48)])
    density = np.arange(384)
    provenance = {
        "version": 3,
        "shanks": {
            "0": {
                "blocks": [
                    {"label": "main", "rows": list(range(384))},
                    {"label": "surface", "rows": list(range(384, 768))},
                ]
            }
        },
    }
    cache = _cache(coords, [density], [0] * 768, provenance=provenance)
    payload = resolve_plot_payload(cache, "line.unit_density")
    np.testing.assert_array_equal(payload["y"], depths)
    np.testing.assert_array_equal(payload["x"], density)


@pytest.mark.parametrize("banks,bins", [(2, 192), (1, 384)])
def test_one_bank_two_bank_mismatch_is_unavailable_with_dimensions(banks, bins, caplog):
    depths = _bank_depths(banks)
    cache = _cache(_paired(depths), np.ones((1, bins)))
    for _ in range(2):
        assert resolve_plot_payload(cache, "line.unit_density") is None
        assert not available_plot_specs_for_menu(cache, "line")
        cache.filter_units("aind_qc")
    assert "_iblqc_ephysRawUnitDensity.density.npy" in caplog.text
    assert f"(1, {bins})" in caplog.text
    assert f"{len(depths)}" in caplog.text
    assert "coverage" in caplog.text
    assert len(caplog.records) == 1


@pytest.mark.parametrize(
    "density",
    [
        np.ones((2, 3)),
        np.ones((3, 1)),
        np.ones(3),
        np.ones((1, 0)),
        np.ones((0, 3)),
        np.ones((1, 3, 1)),
        np.array([["a", "b", "c"]]),
        np.array([[1, 2, np.nan]]),
        np.array([[1, 2, np.inf]]),
        np.array([[1j, 2, 3]]),
        np.array([[True, False, True]]),
    ],
)
def test_invalid_array_never_renders(density, caplog):
    cache = _cache(_paired([0, 20, 40]), density)
    assert resolve_plot_payload(cache, "line.unit_density") is None
    assert not available_plot_specs_for_menu(cache, "line")
    assert "_iblqc_ephysRawUnitDensity.density.npy" in caplog.text


def test_absent_density_keeps_older_stream_usable_without_warning(caplog):
    cache = _cache(_paired([0, 20, 40]), None)
    assert resolve_plot_payload(cache, "line.unit_density") is None
    assert not available_plot_specs_for_menu(cache, "line")
    assert not caplog.records


def test_duplicate_axial_and_duplicate_channel_rows_do_not_duplicate_bins():
    depths = _bank_depths()
    coords = np.vstack([_paired(depths), [[0, 20]]])
    cache = _cache(coords, [np.arange(192)])
    payload = resolve_plot_payload(cache, "line.unit_density")
    np.testing.assert_array_equal(payload["y"], depths)


def test_nonpaired_geometry_does_not_assume_half_channel_count():
    depths = _bank_depths(pitch=15)
    cache = _cache(
        np.column_stack([np.zeros(depths.size), depths]), [np.arange(depths.size)]
    )
    np.testing.assert_array_equal(
        resolve_plot_payload(cache, "line.unit_density")["y"], depths
    )


def test_raw_density_unchanged_by_filter_and_separate_stream_caches():
    depths = _bank_depths()
    cache = _cache(_paired(depths), [np.ones(192)])
    first = resolve_plot_payload(cache, "line.unit_density")
    cache.filter_units("aind_qc")
    second = resolve_plot_payload(cache, "line.unit_density")
    np.testing.assert_array_equal(second["x"], first["x"])
    other_depths = _bank_depths(2)
    other = _cache(_paired(other_depths), [np.ones(384) * 2])
    np.testing.assert_array_equal(
        resolve_plot_payload(other, "line.unit_density")["x"], np.ones(384) * 2
    )
    np.testing.assert_array_equal(
        resolve_plot_payload(other, "line.unit_density")["y"], other_depths
    )


def test_legacy_ambiguous_shank_selection_fails_closed(caplog):
    # The older plot fallback takes two x columns per shank, which is unsafe
    # for streams with one column per shank. Do not reproduce that guess here.
    depths = _bank_depths()
    coords = np.array([[x, y] for x in (0, 250) for y in depths])
    cache = _cache(coords, np.ones((2, 192)), runtime=False)
    assert resolve_plot_payload(cache, "line.unit_density") is None
    assert "geometry" in caplog.text


@pytest.mark.parametrize(
    "depths",
    [
        _bank_depths() + 20,
        _bank_depths() + 3840,
        np.delete(_bank_depths(), 10),
        np.concatenate([_bank_depths(), _bank_depths() + 7680]),
        np.array([0]),
    ],
)
def test_matching_length_cannot_hide_unresolved_producer_bank_axis(depths, caplog):
    cache = _cache(_paired(depths), np.ones((1, len(depths))))
    assert resolve_plot_payload(cache, "line.unit_density") is None
    assert "expected complete ascending zero-origin 3840" in caplog.text


def test_invalid_other_shank_geometry_rejects_entire_array(caplog):
    depths = _bank_depths()
    coords = np.vstack([_paired(depths), _paired(depths + 3840, x=250)])
    cache = _cache(coords, np.ones((2, 192)), [0] * 384 + [1] * 384)
    assert resolve_plot_payload(cache, "line.unit_density") is None
    assert "shank 1" in caplog.text


def test_equal_length_uniform_grid_cannot_hide_reversed_source_banks(caplog):
    depths = _bank_depths(2)
    provenance = {
        "version": 3,
        "shanks": {
            "0": {
                "blocks": [
                    {"label": "surface", "rows": list(range(384))},
                    {"label": "main", "rows": list(range(384, 768))},
                ]
            }
        },
    }
    cache = _cache(_paired(depths), np.ones((1, 384)), provenance=provenance)
    assert resolve_plot_payload(cache, "line.unit_density") is None
    assert "shank main" in caplog.text


def test_overlapping_main_surface_row_provenance_is_ambiguous(caplog):
    provenance = {
        "version": 3,
        "shanks": {
            "0": {
                "blocks": [
                    {"label": "main", "rows": list(range(384))},
                    {"label": "surface", "rows": list(range(384))},
                ]
            }
        },
    }
    cache = _cache(_paired(_bank_depths()), np.ones((1, 192)), provenance=provenance)
    assert resolve_plot_payload(cache, "line.unit_density") is None
    assert "overlapping main/surface" in caplog.text


@pytest.mark.parametrize(
    "provenance",
    [
        None,
        {"version": 2},
        {"version": 3, "shanks": {}},
        {
            "version": 3,
            "shanks": {"0": {"blocks": [{"label": "unknown", "rows": [0]}]}},
        },
        {"version": 3, "shanks": {"0": {"blocks": [{"label": "main", "rows": [-1]}]}}},
        {"version": 3, "shanks": {"0": {"blocks": [{"label": "main", "rows": [384]}]}}},
        {"version": 3, "shanks": {"0": {"blocks": [{"label": "main", "rows": [0.0]}]}}},
        {"version": 3, "shanks": {"0": {"blocks": [{"label": "main", "rows": [0]}]}}},
    ],
)
def test_missing_or_invalid_source_provenance_fails_closed(provenance, caplog):
    cache = _cache(_paired(_bank_depths()), np.ones((1, 192)), provenance=provenance)
    assert resolve_plot_payload(cache, "line.unit_density") is None
    assert "row_channels.json" in caplog.text


def test_runtime_cache_switch_and_invalidation_rebuild_density():
    depths = _bank_depths()
    coords = np.vstack([_paired(depths), _paired(depths, x=250)])
    cache = _cache(
        coords, np.vstack([np.ones(192), np.ones(192) * 2]), [0] * 384 + [1] * 384
    )
    runtime = EphysStreamRuntime(
        cache.channel_collection.stream, EphysPlotPayloadCacheFactory()
    )
    first = runtime.plot_payload_for_shank(0, "line.unit_density")
    second = runtime.plot_payload_for_shank(1, "line.unit_density")
    np.testing.assert_array_equal(first["x"], np.ones(192))
    np.testing.assert_array_equal(second["x"], np.ones(192) * 2)
    assert runtime.plot_payload_for_shank(0, "line.unit_density") is first
    runtime.invalidate_plot_payload_cache()
    assert runtime.plot_payload_for_shank(0, "line.unit_density") is not first
    np.testing.assert_array_equal(
        runtime.plot_payload_for_shank(1, "line.unit_density")["x"], second["x"]
    )
