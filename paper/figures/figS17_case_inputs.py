"""Native-swath case inputs from the canonical registry; no interpolation.

Requires derived/case_display and derived/case_geolocation in the registry.
Reconstructs full 2-D swaths by exact pixel identity. Scientific gaps remain
masked. pcolormesh uses native centers with shading='nearest'; it only constructs
cell edges, never interpolates scientific values. Any nonfinite full-swath
geolocation is an error, including coordinates outside the displayed zoom.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import numpy as np
import pandas as pd

from dbdust.config import CONFIG
from dbdust.modis_db import valid_geolocation
from paper.derive_case_display import CASES
from paper.figures import figstyle as style
from paper.numbers import Registry, digest


FIELDS = ('aod550_db', 'DB_AE', 'branch', 'db_aerosol_type')
TITLES = ('DB total AOD', 'DB Angstrom exponent', 'Golden branch', 'DB aerosol type')
BRANCHES = [(0, '0: No ML', '#DDDDDD'),
            (CONFIG['branches']['full8']['code'], '1: full8', '#332288'),
            (CONFIG['branches']['noSSA5']['code'], '2: noSSA5', '#88CCEE')]
TYPES = [(0, '0: Mixed', '#332288'), (1, '1: Dust', '#DDCC77'),
         (2, '2: Smoke', '#CC6677'), (3, '3: Sulfate', '#117733')]
HEIGHT_MM = 215
ROW_TOP_MM = (197, 131, 65)
ROW_HEADER_MM = (209, 139, 73)


def _integers(series, name):
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Invalid integer geometry column: ' + name)
    return values.astype(np.int64)


def reconstruct(display, geolocation, case):
    """Return native geometry, masked-in-place field grids, and display metadata.

    Requires complete geometry and exactly the zoom's registered science rows;
    neither a nearest-coordinate match nor a dropped/duplicated pixel is allowed.
    Input tables are not modified. Row ordering in either registry table is free.
    """
    d = display.loc[display.case.eq(case)].copy()
    g = geolocation.loc[geolocation.case.eq(case)].copy()
    if d.empty or g.empty:
        raise ValueError('Missing registered case: ' + case)
    shape = []
    for name in ('swath_rows', 'swath_cols'):
        values = _integers(g[name], name)
        if len(np.unique(values)) != 1 or values[0] <= 0 or not (d[name] == values[0]).all():
            raise ValueError('Inconsistent registered swath shape')
        shape.append(int(values[0]))
    shape = tuple(shape)
    g['pixel_index'] = _integers(g.pixel_index, 'pixel_index')
    g = g.sort_values('pixel_index').reset_index(drop=True)
    n = int(np.prod(shape))
    if not np.array_equal(g.pixel_index.to_numpy(), np.arange(n)):
        raise ValueError('Incomplete or duplicated full-swath geometry')
    for name, expected in [('swath_row', np.arange(n) // shape[1]), ('swath_col', np.arange(n) % shape[1])]:
        if not np.array_equal(_integers(g[name], name), expected):
            raise ValueError('Geometry row/column disagrees with pixel_index')
    lat, lon = g.lat.to_numpy(float), g.lon.to_numpy(float)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError('Nonfinite full-swath geolocation: refuse pcolormesh, no coordinate filling')
    if not valid_geolocation(lat, lon).all():
        raise ValueError('Out-of-range full-swath geolocation')
    west, east, south, north = CASES[case][2]
    inside = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    if not g.inside_zoom.isin([True, False]).all() or not np.array_equal(g.inside_zoom.to_numpy(bool), inside):
        raise ValueError('Registered inside_zoom disagrees with prescribed bounds')
    d['pixel_index'] = _integers(d.pixel_index, 'pixel_index')
    d = d.sort_values('pixel_index').reset_index(drop=True)
    indices = d.pixel_index.to_numpy()
    if not np.array_equal(indices, np.flatnonzero(inside)):
        raise ValueError('Case display must contain every zoom pixel exactly once')
    for name in ('lat', 'lon', 'swath_row', 'swath_col'):
        if not np.array_equal(d[name].to_numpy(), g.iloc[indices][name].to_numpy()):
            raise ValueError('Display/geolocation pixel mismatch: ' + name)
    for name, expected in zip(('zoom_west', 'zoom_east', 'zoom_south', 'zoom_north'), (west, east, south, north)):
        if not d[name].eq(expected).all():
            raise ValueError('Case display bounds differ from prescribed zoom')
    date, platform, _ = CASES[case]
    if not d.date.eq(date).all() or not d.platform.eq(platform).all() or d.granule_start_utc.nunique() != 1:
        raise ValueError('Case physical identity differs from figure definition')
    values = {}
    for field in FIELDS:
        grid = np.full(n, np.nan)
        numbers = d[field].to_numpy(float)
        if np.isinf(numbers).any():
            raise ValueError('Infinite registered science value')
        if field in ('branch', 'db_aerosol_type'):
            allowed = [r[0] for r in (BRANCHES if field == 'branch' else TYPES)]
            if not np.isin(numbers[np.isfinite(numbers)], allowed).all():
                raise ValueError('Unrecognized registered category: ' + field)
        if field == 'aod550_db' and (numbers < CONFIG['inputs']['minimum_aod']).any():
            raise ValueError('Negative registered DB total AOD')
        grid[indices] = numbers
        values[field] = grid.reshape(shape)
    return dict(case=case, geometry=g, lat=lat.reshape(shape), lon=lon.reshape(shape),
                values=values, bounds=(west, east, south, north), date=date, platform=platform,
                granule_start_utc=d.granule_start_utc.iloc[0])


def local_coastlines():
    """Load installed coastline assets directly; never invoke a downloader."""
    for key in ('pre_existing_data_dir', 'data_dir'):
        root = Path(cartopy.config[key])
        path = root / 'shapefiles/natural_earth/physical/ne_110m_coastline.shp'
        files = [path.with_suffix(suffix) for suffix in ('.shp', '.shx', '.dbf')]
        if all(p.is_file() for p in files):
            reader = shapereader.Reader(path)
            try:
                geometries = list(reader.geometries())
            finally:
                reader.close()
            return geometries, {str(p.resolve()): digest(p) for p in files}
    raise FileNotFoundError('Local NaturalEarth110m coastline unavailable; no download permitted')


def draw_panel(ax, swath, field):
    """Draw unmodified values using native pixel footprints and masked gaps."""
    if not np.isfinite(swath['lat']).all() or not np.isfinite(swath['lon']).all():
        raise ValueError('Nonfinite swath coordinates: refuse pcolormesh')
    z = swath['values'][field]
    kwargs = dict(shading='nearest', transform=ccrs.PlateCarree(), rasterized=True)
    if field in ('aod550_db', 'DB_AE'):
        levels = np.linspace(0, 4 if field == 'aod550_db' else 2, 11)
        cmap = (style.CMAP_DAOD if field == 'aod550_db' else style.CMAP_P).copy()
        ax.pcolormesh(swath['lon'], swath['lat'], np.ma.masked_where(z != 0, np.zeros_like(z)),
                      cmap=ListedColormap([style.ZERO_GREY]), **kwargs)
        plotted = np.ma.masked_where(~np.isfinite(z) | (z == 0), z)
    else:
        categories = BRANCHES if field == 'branch' else TYPES
        codes = [r[0] for r in categories]
        if codes != list(range(len(codes))):
            raise ValueError('Categorical rendering requires contiguous registered codes')
        levels = np.arange(len(codes) + 1) - .5
        cmap = ListedColormap([r[2] for r in categories])
        plotted = np.ma.masked_invalid(z)
    # Transparent masked layer keeps the explicit grey-zero layer visible.
    cmap.set_bad((1, 1, 1, 0))
    ax.set_facecolor(style.MISSING)
    return ax.pcolormesh(swath['lon'], swath['lat'], plotted, cmap=cmap,
                         norm=BoundaryNorm(levels, cmap.N), **kwargs)


def make(registry_path, output):
    registry = Registry.read(registry_path)
    display = registry.table('derived/case_display')
    geolocation = registry.table('derived/case_geolocation')
    if set(display.case) != set(CASES) or set(geolocation.case) != set(CASES):
        raise ValueError('Registry must contain exactly the three prescribed cases')
    # Validate all cases before creating any plot or output files.
    swaths = [reconstruct(display, geolocation, case) for case in CASES]
    coastlines, geometry_sources = local_coastlines()
    fig = plt.figure(figsize=(style.DOUBLE, HEIGHT_MM * style.MM))
    exported, meshes = [], {}
    try:
        for row, swath in enumerate(swaths):
            name = ('Kansas dust', 'Amazonia smoke', 'North America smoke')[row]
            time = swath['granule_start_utc'].split('T')[1][:5]
            fig.text(.075, ROW_HEADER_MM[row] / HEIGHT_MM, f'{name} | {swath["date"]} | {swath["platform"]} | {time} UTC',
                     fontsize=7, va='top')
            for col, field in enumerate(FIELDS):
                ax = fig.add_axes([.075 + col * .23, (ROW_TOP_MM[row] - 43) / HEIGHT_MM,
                                   .205, 43 / HEIGHT_MM],
                                  projection=ccrs.PlateCarree())
                ax.set_extent(swath['bounds'], crs=ccrs.PlateCarree())
                ax.set_anchor('N')
                meshes[field] = draw_panel(ax, swath, field)
                ax.add_geometries(coastlines, ccrs.PlateCarree(), facecolor='none',
                                  edgecolor='#4D4D4D', linewidth=.4, zorder=3)
                grid = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                   xlocs=np.arange(-180, 181, 5), ylocs=np.arange(-90, 91, 5),
                                   linewidth=.2, color='#BDBDBD')
                grid.top_labels = grid.right_labels = grid.geo_labels = False
                # Each row has different bounds, so longitude labels are required
                # on every row rather than only on the bottom of the whole figure.
                grid.left_labels, grid.bottom_labels = col == 0, True
                grid.xlabel_style = {'size': 6.5, 'rotation': 45, 'ha': 'right'}
                grid.ylabel_style = {'size': 6.5}
                grid.rotate_labels = False
                if row == 0:
                    ax.set_title(TITLES[col], fontsize=7, pad=6)
                style.panel_label(ax, 'abcdefghijkl'[row * 4 + col], x=.02, y=.90)
                ax.texts[-1].set_bbox(dict(facecolor='white', edgecolor='none', pad=.4))
            data = swath['geometry'].copy()
            for field in FIELDS:
                data[field] = swath['values'][field].ravel()
            data = data.assign(date=swath['date'], platform=swath['platform'], granule_start_utc=swath['granule_start_utc'])
            exported.append(data)
        for col, field in enumerate(FIELDS[:2]):
            cax = fig.add_axes([.12 + col * .23, 27 / HEIGHT_MM, .115, 2.5 / HEIGHT_MM])
            bar = fig.colorbar(meshes[field], cax=cax, orientation='horizontal',
                               extend='max' if col == 0 else 'both')
            bar.set_ticks([0, 2, 4] if col == 0 else [0, 1, 2])
            bar.set_label('AOD (550 nm)' if col == 0 else 'Angstrom exponent', fontsize=6.5)
            legend_ax = fig.add_axes([.075 + col * .23, 4 / HEIGHT_MM, .205, 10 / HEIGHT_MM])
            legend_ax.set_axis_off()
            legend_ax.legend(handles=[Patch(facecolor=style.ZERO_GREY, label='Exact zero'),
                                      Patch(facecolor='white', edgecolor='#777777', label='Missing')],
                             loc='center', fontsize=6.5, ncol=1, handlelength=1, borderaxespad=0)
        for col, categories in ((2, BRANCHES), (3, TYPES)):
            legend_ax = fig.add_axes([.075 + col * .23, 4 / HEIGHT_MM, .205, 28 / HEIGHT_MM])
            legend_ax.set_axis_off()
            handles = [Patch(facecolor=color, edgecolor='#555555', linewidth=.3, label=label)
                       for _, label, color in categories]
            handles.append(Patch(facecolor='white', edgecolor='#777777', label='Missing'))
            legend_ax.legend(handles=handles, loc='upper left', fontsize=6.5, handlelength=1,
                             borderaxespad=0, labelspacing=.4)
        evidence = [dict(table=name, source_file=registry.payload['tables'][name]['source_file'],
                         source_sha256=registry.payload['tables'][name]['source_sha256'],
                         derivation=registry.payload['tables'][name]['derivation'])
                    for name in ('derived/case_display', 'derived/case_geolocation')]
        evidence.append(dict(coastline_sources=geometry_sources, rendering='native pcolormesh; shading=nearest; '
                             'no science interpolation, missing-value filling, coordinate repair, land mask or branch reselection',
                             scale_bounds=dict(aod550_db=[0, 4], DB_AE=[0, 2]),
                             categorical_branch=BRANCHES, categorical_aerosol_type=TYPES))
        style.export(fig, output, 'FigS17_case_inputs', pd.concat(exported, ignore_index=True),
                     registry_path, evidence, __file__, raster=True)
        (Path(output) / 'FigS17_caption.md').write_text(
            'Native-swath Deep Blue inputs for the Kansas dust, Amazonia smoke, and North America '
            'smoke development cases. Columns show total AOD, Angstrom exponent, golden branch, '
            'and aerosol type. Original pixels are placed using full golden-swath geolocation; '
            'missing retrievals remain white, without scientific interpolation or gap filling. '
            'Shared color scales are AOD 0-4 and Angstrom exponent 0-2; colorbar extensions '
            'identify out-of-scale values without changing the exported data. Exact zeros in '
            'the continuous fields are grey. Categorical codes and colors are listed below '
            'their columns; no hatching is used. Golden branches are not recomputed from SSA alone. '
            'Times are granule starts. No additional land or confidence mask is applied.\n')
    finally:
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    make(args.registry, args.output_dir)
