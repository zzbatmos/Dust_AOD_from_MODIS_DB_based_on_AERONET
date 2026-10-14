"""Synthetic sign-only display checks; no scientific selection or new analysis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.text import Text
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from paper.figures import difference_encoding as signs
from paper.figures import figstyle as style
from paper.numbers import Registry, digest


def test_exact_sign_no_tolerance_and_masked_missing_zero_infinite():
    tiny = np.nextafter(0., 1.)
    values = np.ma.array([[tiny, -tiny, 0., -0., np.nan],
                          [np.inf, -np.inf, 1., .2, -.2]],
                         mask=[[False] * 5, [False, False, True, False, False]])
    original = values.copy()
    np.testing.assert_array_equal(signs.positive_mask(values),
                                  [[True, False, False, False, False],
                                   [False, False, False, True, False]])
    np.testing.assert_array_equal(values.data, original.data)
    np.testing.assert_array_equal(values.mask, original.mask)
    with pytest.raises(ValueError, match="two-dimensional"):
        signs.positive_mask([1, 2])


@pytest.mark.parametrize("projection", [ccrs.PlateCarree(), ccrs.Robinson(), ccrs.Mollweide()])
@pytest.mark.parametrize("geometry", ["edges", "swath", "dateline"])
def test_exact_native_polygons_projection_wrap_and_base_unchanged(projection, geometry):
    z = np.array([[.1, -.1, np.nan], [0., .05, -.05]])
    if geometry == "edges":
        x, y = [-10, -5, 0, 5], [-4, 0, 4]
    else:
        x, y = np.meshgrid([177., 179., 181.] if geometry == "dateline" else [-8., -4., 0.], [-2., 2.])
        x = x + y * .1
        if geometry == "dateline":
            x = (x + 180) % 360 - 180
    fig = plt.figure()
    ax = fig.add_subplot(projection=projection)
    ax.set_global()
    norm = BoundaryNorm(np.linspace(-.15, .15, 11), style.CMAP_DIFF.N)
    mesh = ax.pcolormesh(x, y, np.ma.masked_invalid(z), shading="auto", norm=norm,
                         cmap=style.CMAP_DIFF, transform=ccrs.PlateCarree())
    coordinates = mesh.get_coordinates().copy()
    base = mesh.get_array().copy()
    artists = signs.add_positive_stipple(ax, mesh, z)
    assert len(artists) == 2
    for artist in artists:
        assert len(artist.get_paths()) == 2
        assert artist.get_rasterized() and artist.get_zorder() < 3
        assert artist.get_array() is None and artist.get_facecolors().size == 0
        assert np.all(artist.get_linewidths() == 0)
        assert artist.get_hatch_linewidth() == .25
        for path, (i, j) in zip(artist.get_paths(), np.argwhere(signs.positive_mask(z))):
            expected = coordinates[[i, i, i + 1, i + 1], [j, j + 1, j + 1, j]]
            assert set(map(tuple, path.vertices)) == set(map(tuple, expected))
    assert artists[0].get_hatch() == signs.HALO_HATCH
    assert artists[1].get_hatch() == signs.DOT_HATCH
    np.testing.assert_array_equal(mesh.get_coordinates(), coordinates)
    np.testing.assert_array_equal(mesh.get_array(), base)
    assert mesh.norm is norm and mesh.cmap is style.CMAP_DIFF
    fig.canvas.draw()  # Actual projected drawing, including dateline polygons.
    plt.close(fig)


def test_stipple_visible_on_light_and_dark_but_never_zero_negative_or_missing():
    fig = plt.figure(figsize=(6, 1), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent([0, 6, 0, 1], crs=ccrs.PlateCarree())
    ax.set_axis_off()
    z = np.array([[-.15, 0., np.nan, np.inf, np.nextafter(0., 1.), .15]])
    mesh = ax.pcolormesh(np.arange(7), [0, 1], np.ma.masked_invalid(z),
                         cmap=style.CMAP_DIFF, vmin=-.15, vmax=.15, transform=ccrs.PlateCarree())
    fig.canvas.draw()
    before = np.array(fig.canvas.buffer_rgba())
    signs.add_positive_stipple(ax, mesh, z)
    fig.canvas.draw()
    after = np.array(fig.canvas.buffer_rgba())
    assert np.array_equal(before[:, :598], after[:, :598])
    for left in (600, 750):
        tile = after[10:-10, left + 10:left + 140, :3]
        assert (tile.min(axis=2) < 30).any()  # Black dot.
        assert (tile.min(axis=2) > 240).any()  # White halo, even on dark red.
        assert not np.array_equal(before[10:-10, left + 10:left + 140],
                                  after[10:-10, left + 10:left + 140])
    plt.close(fig)


def test_empty_mask_and_invalid_geometry_fail_closed():
    class Mesh:
        def get_coordinates(self):
            return np.zeros((3, 3, 2))
    class NoDraw:
        def pcolor(self, *args, **kwargs):
            pytest.fail("No overlay should be drawn")
    assert signs.add_positive_stipple(NoDraw(), Mesh(), np.zeros((2, 2))) == ()
    with pytest.raises(ValueError, match="edges"):
        signs.add_positive_stipple(NoDraw(), Mesh(), np.ones((3, 3)))
    class BadMesh:
        def get_coordinates(self):
            return np.full((3, 3, 2), np.nan)
    with pytest.raises(ValueError, match="edges"):
        signs.add_positive_stipple(NoDraw(), BadMesh(), np.ones((2, 2)))


def test_legend_and_helper_hash_are_explicit():
    handles = signs.sign_handles()
    assert handles[0].get_hatch() != handles[1].get_hatch() == "////"
    assert handles[0].get_label() == ">0 (sign only)"
    provenance = signs.provenance()
    assert "not significance" in provenance["display_encoding"]
    assert provenance["encoding_sha256"] == digest(Path(signs.__file__))


@pytest.fixture
def no_online_geography(monkeypatch):
    monkeypatch.setattr(GeoAxes, "coastlines", lambda *a, **k: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *a, **k: None)


def registry_for(tables):
    return Registry({"tables": {name: dict(records=frame.to_dict("records"), columns=list(frame),
                                           source_file="synthetic.csv", source_sha256="synthetic", derivation={})
                                 for name, frame in tables.items()}})


def check_export(fig, output, stem, frame, registry_path, evidence, script, raster=False):
    fig.canvas.draw()
    assert raster and fig.get_figwidth() / style.MM == pytest.approx(190)
    assert sum(a.get_gid() == "positive-difference-black"
               for ax in fig.axes for a in ax.collections) == (4 if stem.startswith("Fig06") else 3)
    assert any(e.get("encoding_sha256") == digest(Path(signs.__file__)) for e in evidence)
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontsize() >= 6.5
            box = text.get_window_extent(fig.canvas.get_renderer())
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    Path(output).mkdir(exist_ok=True)


def test_fig6_calls_native_sign_overlay_and_caption_bound(monkeypatch, tmp_path, no_online_geography):
    from paper.figures import fig06_seasonal_maps as renderer
    rows = []
    for period in ("DJF", "MAM", "JJA", "SON"):
        for i, lat in enumerate([-2, 2]):
            for j, lon in enumerate([-5, 0, 5]):
                value = [.05, -.05, np.nan][j]
                rows.append(dict(period=period, lat=lat, lon=lon, lat_index=i, lon_index=j,
                                 lon_lower=lon - 2.5, lon_upper=lon + 2.5, lat_lower=lat - 2,
                                 lat_upper=lat + 2, ev=abs(value), difference=value,
                                 supported=j != 2, in_domain=True, has_land=True))
    data = pd.DataFrame(rows)
    boxes = pd.DataFrame([dict(region=r, lon_lower=0, lon_upper=1, lat_lower=0, lat_upper=1)
                          for r in renderer.REGIONS])
    registry = registry_for({"derived/seasonal_maps": data, "derived/seasonal_region_boxes": boxes,
                             "derived/seasonal_map_metrics": pd.DataFrame()})
    monkeypatch.setattr(renderer.Registry, "read", lambda _: registry)
    monkeypatch.setattr(renderer.style, "export", check_export)
    renderer.make(tmp_path / "unused.json", tmp_path / "out")
    caption = (tmp_path / "out/Fig06_caption.md").read_text()
    assert len(caption.split()) <= 80
    assert "strictly positive" in caption and "not significance" in caption
    pd.testing.assert_frame_equal(data, registry.table("derived/seasonal_maps"))


def test_fig8_native_overlay_halo_and_caption_bound(monkeypatch, tmp_path, no_online_geography):
    from paper.figures import fig08_cases as renderer
    rows, geo = [], []
    for case in renderer.CASES:
        image = tmp_path / (case + "_GIBS_truecolor.png")
        Image.new("RGB", (10, 10), "gray").save(image)
        image.with_suffix(".json").write_text(json.dumps(dict(sha256=digest(image), platform="Test",
                                                            date="2000-01-01", extent_lonlat=[-10, 10, -10, 10])))
        for i, lat in enumerate([-5., 0., 5.]):
            for j, lon in enumerate([-5., 0., 5.]):
                geo.append(dict(case=case, pixel_index=i * 3 + j, swath_rows=3, swath_cols=3, lat=lat, lon=lon))
                rows.append(dict(geo[-1], date="2000-01-01", platform="Test", granule_start_utc="2000-01-01T00:00Z",
                                 zoom_west=-10, zoom_east=10, zoom_south=-10, zoom_north=10,
                                 dust_probability=.2, daod550_qa2=.1, daod550_ev=.1, lg25_daod=.1,
                                 ev_minus_lg25=[-.1, 0., .1][j], difference_missing=0, full8_branch=float(j == 2)))
    registry = registry_for({"derived/case_display": pd.DataFrame(rows), "derived/case_geolocation": pd.DataFrame(geo)})
    monkeypatch.setattr(renderer.Registry, "read", lambda _: registry)
    def export(*args, **kwargs):
        from matplotlib.contour import ContourSet
        check_export(*args, **kwargs)
        contours = [a for ax in args[0].axes for a in ax.collections if isinstance(a, ContourSet)]
        assert len(contours) == 3
        for contour in contours:
            np.testing.assert_array_equal(contour.levels, [.5])
            assert len(contour.get_path_effects()) == 2
    monkeypatch.setattr(renderer.style, "export", export)
    renderer.make(tmp_path / "unused.json", tmp_path, tmp_path / "out")
    caption = (tmp_path / "out/Fig08_caption.md").read_text()
    assert len(caption.split()) <= 80
    assert "not significance" in caption and "White-haloed black contours" in caption


def test_s13_difference_only_page_with_exact_native_mask(monkeypatch, tmp_path):
    from paper.figures import figS13_all_seasonal_maps as renderer
    grid = dict(x=np.array([-10., 0., 10.]), y=np.array([-10., 0., 10.]),
                value=np.ones((2, 2)), difference=np.array([[.1, -.1], [0., np.nan]]),
                missing_land=np.array([[False, False], [False, True]]))
    grids = {(method, season): grid for method in renderer.METHODS for season in renderer.SEASONS}
    data = pd.DataFrame({"estimator": renderer.METHODS})
    monkeypatch.setattr(renderer, "select", lambda _: (data, grids, []))
    monkeypatch.setattr(renderer.Registry, "read", lambda _: None)
    monkeypatch.setattr(renderer, "local_geography", lambda _: ({"ocean": (), "coastline": ()}, {}))
    seen = []
    def export(fig, output, stem, exported, registry_path, evidence, script, raster):
        renderer.check_layout(fig)
        assert stem == "FigS13b_all_seasonal_differences"
        assert not exported.estimator.eq("CALIOP").any()
        assert len(fig.axes) == 29
        assert sum(a.get_gid() == "positive-difference-black" for ax in fig.axes for a in ax.collections) == 28
        assert evidence[-1] == signs.provenance()
        seen.append(stem)
        Path(output).mkdir()
    monkeypatch.setattr(renderer.style, "export", export)
    path = tmp_path / "registry.json"
    path.write_text("synthetic")
    renderer.make(path, tmp_path / "out", page="difference")
    assert len(seen) == 1 and not list((tmp_path / "out").glob("FigS13a*"))
    assert "not significance" in (tmp_path / "out/FigS13b_caption.md").read_text()
    with pytest.raises(ValueError, match="page"):
        renderer.make(path, tmp_path / "unused", page="unknown")
