"""Tests for raw activity-density line payloads."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ephys_alignment_gui.plotting.builders.unit_density import (
    UnitDensityPlotDataBuilder,
)


def _geometry(*, pitch_um: float = 20.0):
    return SimpleNamespace(chn_diff=pitch_um)


def test_build_selects_shank_and_preserves_depth_row_positions() -> None:
    data = {
        "raw_unit_density": {
            "exists": True,
            "density": np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [0.0, np.nan, 5.0, 2.0],
                ]
            ),
        }
    }

    payload = UnitDensityPlotDataBuilder(data, _geometry(), shank_idx=1).build()

    np.testing.assert_array_equal(payload["x"], np.array([0.0, 5.0, 2.0]))
    np.testing.assert_array_equal(payload["y"], np.array([0.0, 40.0, 60.0]))
    np.testing.assert_array_equal(payload["xrange"], np.array([0.0, 5.0]))
    assert payload["xaxis"] == "Normalized Unit Density (mm⁻¹)"


def test_build_returns_none_when_density_is_unavailable() -> None:
    payload = UnitDensityPlotDataBuilder(
        {"raw_unit_density": {"exists": False}},
        _geometry(),
        shank_idx=0,
    ).build()

    assert payload is None


def test_build_rejects_missing_shank_row() -> None:
    builder = UnitDensityPlotDataBuilder(
        {
            "raw_unit_density": {
                "exists": True,
                "density": np.ones((1, 4)),
            }
        },
        _geometry(),
        shank_idx=1,
    )

    with pytest.raises(ValueError, match="cannot select shank 1"):
        builder.build()
