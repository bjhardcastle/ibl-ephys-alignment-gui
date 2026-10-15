"""Line-plot payload builder for producer-estimated raw activity density."""

from __future__ import annotations

import numpy as np

from ephys_alignment_gui.plotting.channel_geometry import PlotChannelGeometry


class UnitDensityPlotDataBuilder:
    """Build a shank-scoped normalized unit-density depth profile."""

    def __init__(
        self,
        data,
        geometry: PlotChannelGeometry,
        shank_idx: int,
    ) -> None:
        self.data = data
        self.geometry = geometry
        self.shank_idx = shank_idx

    def build(self):
        """Return a line payload on the producer's complete depth-row grid."""
        entry = self.data.get("raw_unit_density", {"exists": False})
        if not entry.get("exists", False):
            return None

        density = np.asarray(entry["density"])
        if density.ndim != 2:
            raise ValueError(
                "raw unit density must have shape (shanks, depth rows), "
                f"got {density.shape}"
            )
        if self.shank_idx >= density.shape[0]:
            raise ValueError(
                f"raw unit density has {density.shape[0]} shanks; "
                f"cannot select shank {self.shank_idx}"
            )

        profile = np.asarray(density[self.shank_idx], dtype=float)
        finite = np.isfinite(profile)
        if not np.any(finite):
            return None

        profile = profile[finite]
        depths_um = np.flatnonzero(finite).astype(float) * self.geometry.chn_diff
        maximum = float(np.max(profile))
        return {
            "x": profile,
            "y": depths_um,
            "xrange": np.array([0.0, maximum if maximum > 0 else 1.0]),
            "xaxis": "Normalized Unit Density (mm⁻¹)",
        }
