"""Physical-depth profiles for optional raw unit-density data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry
from ephys_alignment_gui.services.ephys_data import ChannelCollectionView

logger = logging.getLogger(__name__)

# Producer UnitDensityParams.bank_span_um; no alternate span/axis is advertised
# by the datapackage. See converter commit 7731283, unit_density.py:26-42,298-338.
_DENSITY_BANK_SPAN_UM = 3840.0


class UnitDensityPlotDataBuilder:
    """Validate full-stream coverage before exposing a shank's raw profile."""

    def __init__(
        self,
        data: dict[str, Any],
        geometry: PlotChannelGeometry,
        shank_idx: int,
        channel_collection: ChannelCollectionView | None,
    ) -> None:
        self._payload = None
        entry = data.get("raw_unit_density")
        if not entry or not entry.get("exists"):
            return
        try:
            self._payload = self._build(entry, geometry, shank_idx, channel_collection)
        except (ValueError, TypeError, IndexError) as exc:
            diagnostic = str(exc)
            # Stream-owned input is shared by shank caches and filter rebuilds.
            if entry.get("_geometry_error") != diagnostic:
                entry["_geometry_error"] = diagnostic
                logger.warning(
                    "Unit density unavailable at %s: %s",
                    entry.get("path", "<unknown density file>"),
                    diagnostic,
                )

    @property
    def available(self) -> bool:
        """Whether density and physical depth coverage match for the stream."""
        return self._payload is not None

    def build(self) -> dict[str, Any] | None:
        """Return the validated line payload, independent of sorted-unit filters."""
        return self._payload

    @staticmethod
    def _build(entry, geometry, shank_idx, channel_collection):
        density = np.asarray(entry.get("density"))
        if (
            density.ndim != 2
            or density.size == 0
            or density.dtype.kind not in "iuf"
            or not np.all(np.isfinite(density))
        ):
            raise ValueError(
                f"observed shape {density.shape}, dtype {density.dtype}; expected "
                "nonempty finite real numeric (n_shanks, n_bins)"
            )

        coords = np.asarray(geometry.chn_coords_all)
        if coords.dtype.kind not in "iuf" or not np.all(np.isfinite(coords)):
            raise ValueError(
                "channel geometry must contain finite real physical coordinates"
            )

        if channel_collection is not None:
            table = channel_collection.channel_table
            rows_by_shank = [table.rows_for_shank(idx) for idx in range(table.n_shanks)]
            if channel_collection.shank_idx != shank_idx:
                raise ValueError(
                    "active shank does not match channel collection geometry"
                )
        elif geometry.chn_shank_ind_all is not None:
            shanks = geometry.chn_shank_ind_all
            rows_by_shank = [np.flatnonzero(shanks == idx) for idx in np.unique(shanks)]
        else:
            # Legacy shanks are separated by lateral gaps, not a fixed number
            # of columns; columns can vary between probe types and banks.
            xs = np.unique(coords[:, 0])
            groups = np.split(xs, np.flatnonzero(np.diff(xs) > 100) + 1)
            rows_by_shank = [
                np.flatnonzero(np.isin(coords[:, 0], group)) for group in groups
            ]

        n_shanks = len(rows_by_shank)
        depths_by_shank = [np.unique(coords[rows, 1]) for rows in rows_by_shank]
        counts = [depths.size for depths in depths_by_shank]
        if density.shape[0] != n_shanks or any(
            count == 0 or count != density.shape[1] for count in counts
        ):
            coverage = [
                (float(depths[0]), float(depths[-1])) if depths.size else None
                for depths in depths_by_shank
            ]
            raise ValueError(
                f"observed shape {density.shape}; expected {n_shanks} stream shanks "
                f"with axial bin counts {counts}, coverage (µm) {coverage}. "
                "Density must cover every unique physical axial position across "
                "all recorded banks; regenerate density for this stream's channel table."
            )
        if not 0 <= shank_idx < n_shanks:
            raise ValueError(f"active shank {shank_idx} outside stream geometry")

        for idx, shank_depths in enumerate(depths_by_shank):
            _validate_producer_depths(shank_depths, density.shape, idx)

        depths = depths_by_shank[shank_idx]
        expected_coords = np.unique(coords[rows_by_shank[shank_idx]], axis=0)
        if not np.array_equal(expected_coords, np.unique(geometry.chn_coords, axis=0)):
            raise ValueError(
                "active plot geometry differs from stream shank coverage; "
                "supply explicit channels.shankInd.npy"
            )
        _validate_source_order(entry, coords, rows_by_shank, depths_by_shank)
        values = density[shank_idx]
        return {
            "x": values,
            "y": depths,
            "xrange": np.array(
                [min(0.0, float(values.min())), max(1.0, float(values.max()))]
            ),
            "xaxis": "Unit density",
        }


def _validate_producer_depths(depths, shape, shank_idx):
    """Require observed depths to prove correspondence to the producer axis."""
    # compose_raw_unit_density stacks complete main then surface banks and writes
    # rows in ascending zero-origin depth (j * pitch). Missing rows and discarded
    # bank anchors mean even equal lengths cannot justify an arbitrary depth map.
    spacing = np.diff(depths)
    pitch = float(np.min(spacing)) if spacing.size else 0.0
    rows_per_bank = _DENSITY_BANK_SPAN_UM / pitch if pitch > 0 else 0.0
    complete_banks = (
        rows_per_bank >= 1
        and np.isclose(rows_per_bank, round(rows_per_bank))
        and depths.size % round(rows_per_bank) == 0
    )
    if (
        not np.isclose(depths[0], 0.0)
        or not np.allclose(spacing, pitch)
        or not complete_banks
    ):
        raise ValueError(
            f"observed shape {shape}, shank {shank_idx} axial coverage "
            f"{depths[0]:g}..{depths[-1]:g} µm ({depths.size} positions); "
            "expected complete ascending zero-origin 3840 µm bank geometry "
            "at the physical channel-row pitch. Offset, gapped or partial banks "
            "cannot be matched to the producer's density axis; check the "
            "channel table and regenerate density with matching bank coverage."
        )


def _validate_source_order(entry, coords, rows_by_shank, depths_by_shank):
    """Prove main/surface ordering using producer block-to-channel provenance."""
    metadata = entry.get("row_channels")
    metadata_path = entry.get("row_channels_path", "band_corr/row_channels.json")
    if not isinstance(metadata, dict) or metadata.get("version") != 3:
        raise ValueError(
            f"source geometry unresolved: expected version 3 block provenance at "
            f"{metadata_path}; {entry.get('provenance_error', 'metadata missing or incompatible')}"
        )
    shanks = metadata.get("shanks")
    if not isinstance(shanks, dict) or len(shanks) != len(rows_by_shank):
        raise ValueError(
            f"stream shank geometry differs from provenance at {metadata_path}"
        )
    try:
        ordered_keys = sorted(shanks, key=int)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid shank identities at {metadata_path}") from exc
    for idx, key in enumerate(ordered_keys):
        shank = shanks[key]
        if not isinstance(shank, dict):
            raise ValueError(f"invalid shank block geometry at {metadata_path}")
        _validate_shank_sources(
            shank.get("blocks"),
            coords,
            rows_by_shank[idx],
            depths_by_shank[idx],
            metadata_path,
        )


def _validate_shank_sources(blocks, coords, rows, depths, metadata_path):
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"missing main/surface block geometry at {metadata_path}")
    sources = {"main": set(), "surface": set()}
    pitches = []
    allowed_rows = set(rows.tolist())
    for block in blocks:
        if not isinstance(block, dict) or block.get("label") not in sources:
            raise ValueError(f"unknown density block source at {metadata_path}")
        block_rows = np.asarray(block.get("rows"))
        if (
            block_rows.ndim != 1
            or block_rows.size == 0
            or block_rows.dtype.kind not in "iu"
            or not set(block_rows.tolist()).issubset(allowed_rows)
        ):
            raise ValueError(f"invalid source channel rows at {metadata_path}")
        sources[block["label"]].update(block_rows.tolist())
        spacing = np.diff(np.unique(coords[block_rows, 1]))
        if spacing.size:
            pitches.append(float(spacing.min()))
    if sources["main"] | sources["surface"] != allowed_rows or not pitches:
        raise ValueError(f"incomplete source channel coverage at {metadata_path}")
    pitch = min(pitches)
    if not np.isclose(pitch, np.diff(depths).min()):
        raise ValueError(
            f"block row pitch differs from stream geometry at {metadata_path}"
        )
    offset = 0.0
    for label, source_rows in sources.items():
        if not source_rows:
            continue
        source_depths = np.unique(coords[sorted(source_rows), 1])
        _validate_producer_depths(source_depths - offset, (1, depths.size), label)
        offset += source_depths.size * pitch
