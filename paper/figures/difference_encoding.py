"""Display-only sign texture clipped to existing native difference-cell geometry."""
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from paper.numbers import digest


DOT_HATCH = "...."
HALO_HATCH = "oooo"
SIGN_LABEL = ">0 (sign only)"


def positive_mask(values):
    """No magnitude cutoff: masked, missing, infinite and exact-zero cells are off."""
    values = np.ma.asarray(values)
    if values.ndim != 2:
        raise ValueError("Difference values must be a two-dimensional native grid")
    raw = np.ma.getdata(values)
    return ~np.ma.getmaskarray(values) & np.isfinite(raw) & (raw > 0)


def add_positive_stipple(ax, mesh, values, transform=None):
    """Reuse the base QuadMesh edges, including its curvilinear/wrapped geometry.

    Hatch spacing is in display units, not a resampled scientific grid. Every
    hatch fragment is clipped to a finite positive native cell. White rings
    behind black dots retain contrast at both ends of the unchanged color scale.
    """
    selected = positive_mask(values)
    edges = np.asarray(mesh.get_coordinates())
    expected = (selected.shape[0] + 1, selected.shape[1] + 1, 2)
    if edges.shape != expected or not np.isfinite(edges).all():
        raise ValueError("Stipple requires finite base-mesh edges matching the native grid")
    if not selected.any():
        return ()
    transform = ccrs.PlateCarree() if transform is None else transform
    corners = np.stack((edges[:-1, :-1], edges[:-1, 1:],
                        edges[1:, 1:], edges[1:, :-1]), axis=2)[selected]
    artists = []
    for hatch, color in ((HALO_HATCH, "white"), (DOT_HATCH, "black")):
        # Only positive polygons exist in the overlay. Clearing a PolyQuadMesh
        # scalar array would also clear its mask and incorrectly hatch all cells.
        artist = PolyCollection(corners, facecolors="none", hatch=hatch,
                                edgecolors=color, linewidth=0, transform=transform,
                                rasterized=True, zorder=2)
        artist.set_hatch_linewidth(.25)
        artist.set_gid("positive-difference-" + color)
        ax.add_collection(artist, autolim=False)
        artists.append(artist)
    return tuple(artists)


def sign_handles():
    return [Patch(facecolor="white", edgecolor="black", hatch=DOT_HATCH,
                  linewidth=0, label=SIGN_LABEL),
            Patch(facecolor="#E0E0E0", edgecolor="#BDBDBD", hatch="////",
                  linewidth=0, label="Missing")]


def provenance():
    return dict(display_encoding="finite unmasked difference > 0: black dots with white halos; sign only, not significance",
                geometry="existing base QuadMesh cell edges; no interpolation or science-array changes",
                encoding_file=str(Path(__file__).resolve()), encoding_sha256=digest(Path(__file__)))
