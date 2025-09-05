import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal
import json
from pathlib import Path

from .spg_file import SPGFile

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
# THRESHOLDS = [0.06, 0.12, 0.3]
ISOLINES = [0.12]  # 12cm and 20cm bands


# region Compare Geodetic Shifts
def compare_geodetic_shifts(SPG, other_SPG: str, mode: Literal['x', 'y', 'both'] = 'both', saveas: Optional[str] = None):

    n1 = np.linspace(SPG.data["grids"]["geodetic_shifts"]["metadata"]['minn'], SPG.data["grids"]["geodetic_shifts"]["metadata"]['maxn'], SPG.data["grids"]["geodetic_shifts"]["metadata"]['nrows'])
    e1 = np.linspace(SPG.data["grids"]["geodetic_shifts"]["metadata"]['mine'], SPG.data["grids"]["geodetic_shifts"]["metadata"]['maxe'], SPG.data["grids"]["geodetic_shifts"]["metadata"]['ncols'])
    n2 = np.linspace(other_SPG.data["grids"]["geodetic_shifts"]["metadata"]['minn'], other_SPG.data["grids"]["geodetic_shifts"]["metadata"]['maxn'], other_SPG.data["grids"]["geodetic_shifts"]["metadata"]['nrows'])
    e2 = np.linspace(other_SPG.data["grids"]["geodetic_shifts"]["metadata"]['mine'], other_SPG.data["grids"]["geodetic_shifts"]["metadata"]['maxe'], other_SPG.data["grids"]["geodetic_shifts"]["metadata"]['ncols'])

    min_n, max_n = max(n1[0], n2[0]), min(n1[-1], n2[-1])
    min_e, max_e = max(e1[0], e2[0]), min(e1[-1], e2[-1])

    def get_idx(axis, minv, maxv):
        return np.where((axis >= minv) & (axis <= maxv))[0]

    idx_n1 = get_idx(n1, min_n, max_n)
    idx_e1 = get_idx(e1, min_e, max_e)
    idx_n2 = get_idx(n2, min_n, max_n)
    idx_e2 = get_idx(e2, min_e, max_e)

    n = min(len(idx_n1), len(idx_n2))
    e = min(len(idx_e1), len(idx_e2))
    idx_n1, idx_e1 = idx_n1[:n], idx_e1[:e]
    idx_n2, idx_e2 = idx_n2[:n], idx_e2[:e]

    sub1 =       SPG.data["grids"]["geodetic_shifts"]["grid"][:, idx_n1[:, None], idx_e1]
    sub2 = other_SPG.data["grids"]["geodetic_shifts"]["grid"][:, idx_n2[:, None], idx_e2]

    if mode == 'x':
        diff = sub1[0] - sub2[0]
        rmse = np.sqrt(np.nanmean(diff**2))
        title = f'X Shift Difference (RMSE: {rmse:.4f} m)'
    elif mode == 'y':
        diff = sub1[1] - sub2[1]
        rmse = np.sqrt(np.nanmean(diff**2))
        title = f'Y Shift Difference (RMSE: {rmse:.4f} m)'
    elif mode == 'both':
        dx = sub1[0] - sub2[0]
        dy = sub1[1] - sub2[1]
        diff = np.sqrt(dx**2 + dy**2)
        rmse = np.sqrt(np.nanmean(diff**2))
        title = f'Total Shift Difference Magnitude (Vector RMSE: {rmse:.4f} m)'
    else:
        raise ValueError("Mode must be 'x', 'y', or 'both'")

    sub_n = n1[idx_n1]
    sub_e = e1[idx_e1]
    lon_grid, lat_grid = np.meshgrid(sub_e, sub_n)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(lon_grid, lat_grid, diff, shading='auto', cmap='coolwarm')
    plt.colorbar(label='Shift Difference (m)')
    plt.title(title)
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.tight_layout()

    if saveas:
        plt.savefig(saveas)
        plt.close()
    else:
        plt.show()

# endregion

# region Compare Geoid Heights
def compare_geoid_heights(SPG, other_SPG: str, saveas: Optional[str] = None):
    from pathlib import Path
    import geopandas as gpd
    from shapely.geometry import Point
    

    # Load Romania polygon from GeoJSON
    romania_geojson = Path(__file__).parent / 'romania_polygon.geojson'
    romania_gdf = gpd.read_file(romania_geojson)
    romania_poly = romania_gdf.geometry.unary_union

    lat1 = np.linspace(SPG.data["grids"]["geoid_heights"]["metadata"]['minphi'], SPG.data["grids"]["geoid_heights"]["metadata"]['maxphi'], SPG.data["grids"]["geoid_heights"]["metadata"]['nrows'])
    lon1 = np.linspace(SPG.data["grids"]["geoid_heights"]["metadata"]['minla'],  SPG.data["grids"]["geoid_heights"]["metadata"]['maxla'],  SPG.data["grids"]["geoid_heights"]["metadata"]['ncols'])

    lat2 = np.linspace(other_SPG.data["grids"]["geoid_heights"]["metadata"]['minphi'], other_SPG.data["grids"]["geoid_heights"]["metadata"]['maxphi'], other_SPG.data["grids"]["geoid_heights"]["metadata"]['nrows'])
    lon2 = np.linspace(other_SPG.data["grids"]["geoid_heights"]["metadata"]['minla'],  other_SPG.data["grids"]["geoid_heights"]["metadata"]['maxla'],  other_SPG.data["grids"]["geoid_heights"]["metadata"]['ncols'])

    min_lat, max_lat = max(lat1[0], lat2[0]), min(lat1[-1], lat2[-1])
    min_lon, max_lon = max(lon1[0], lon2[0]), min(lon1[-1], lon2[-1])

    def get_idx(axis, minv, maxv):
        return np.where((axis >= minv) & (axis <= maxv))[0]

    idx_lat1 = get_idx(lat1, min_lat, max_lat)
    idx_lon1 = get_idx(lon1, min_lon, max_lon)
    idx_lat2 = get_idx(lat2, min_lat, max_lat)
    idx_lon2 = get_idx(lon2, min_lon, max_lon)

    n = min(len(idx_lat1), len(idx_lat2))
    e = min(len(idx_lon1), len(idx_lon2))
    idx_lat1, idx_lon1 = idx_lat1[:n], idx_lon1[:e]
    idx_lat2, idx_lon2 = idx_lat2[:n], idx_lon2[:e]

    sub1 = np.squeeze(SPG.data["grids"]["geoid_heights"]["grid"])[np.ix_(idx_lat1, idx_lon1)]
    sub2 = np.squeeze(other_SPG.data["grids"]["geoid_heights"]["grid"])[np.ix_(idx_lat2, idx_lon2)]

    lon_grid, lat_grid = np.meshgrid(lon1[idx_lon1], lat1[idx_lat1])

    # Mask points outside Romania
    points = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=-1)
    mask = np.array([romania_poly.contains(Point(lon, lat)) for lon, lat in points])
    mask = mask.reshape(lon_grid.shape)

    diff = sub1 - sub2
    diff_masked = np.where(mask, diff, np.nan)
    rmse = np.sqrt(np.nanmean(diff_masked ** 2))
    mean = np.nanmean(diff_masked)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(lon_grid, lat_grid, diff_masked, shading='auto', cmap='bwr')
    plt.colorbar(label='Geoid Height Difference (m)')
    plt.title(f'Geoid Height Difference (Romania only)\nRMSE: {rmse:.4f} m | Mean: {mean:.4f} m')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    if saveas:
        plt.savefig(saveas)
        plt.close()
    else:
        plt.show()

    return rmse, mean

def compare_geoid_heights_interpolated_nothresholds(SPG, other_SPG: str, saveas: Optional[str] = None):
    from scipy.interpolate import RegularGridInterpolator
    import geopandas as gpd
    from shapely.geometry import Point
    from pathlib import Path

    # Load Romania polygon from GeoJSON
    romania_geojson = Path(__file__).parent / 'romania_polygon.geojson'
    romania_gdf = gpd.read_file(romania_geojson)
    romania_poly = romania_gdf.geometry.unary_union

    # Target grid (SPG)
    lat1 = np.linspace(SPG.data["grids"]["geoid_heights"]["metadata"]['minphi'], SPG.data["grids"]["geoid_heights"]["metadata"]['maxphi'], SPG.data["grids"]["geoid_heights"]["metadata"]['nrows'])
    lon1 = np.linspace(SPG.data["grids"]["geoid_heights"]["metadata"]['minla'],  SPG.data["grids"]["geoid_heights"]["metadata"]['maxla'],  SPG.data["grids"]["geoid_heights"]["metadata"]['ncols'])

    # Source grid (other)
    lat2 = np.linspace(other_SPG.data["grids"]["geoid_heights"]["metadata"]['minphi'], other_SPG.data["grids"]["geoid_heights"]["metadata"]['maxphi'], other_SPG.data["grids"]["geoid_heights"]["metadata"]['nrows'])
    lon2 = np.linspace(other_SPG.data["grids"]["geoid_heights"]["metadata"]['minla'],  other_SPG.data["grids"]["geoid_heights"]["metadata"]['maxla'],  other_SPG.data["grids"]["geoid_heights"]["metadata"]['ncols'])

    grid2 = np.squeeze(other_SPG.data["grids"]["geoid_heights"]["grid"])
    interpolator = RegularGridInterpolator((lat2, lon2), grid2, bounds_error=False, fill_value=np.nan)

    # Generate coordinate pairs for interpolation
    lon_grid, lat_grid = np.meshgrid(lon1, lat1)
    points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)

    # Mask points outside Romania
    mask = np.array([romania_poly.contains(Point(lon, lat)) for lat, lon in points])
    interp_grid2 = interpolator(points)
    interp_grid2[~mask] = np.nan
    interp_grid2 = interp_grid2.reshape(len(lat1), len(lon1))

    # Compute difference
    grid1 = np.squeeze(SPG.data["grids"]["geoid_heights"]["grid"])
    grid1_masked = grid1.copy()
    grid1_masked = grid1_masked.reshape(-1)
    grid1_masked[~mask] = np.nan
    grid1_masked = grid1_masked.reshape(len(lat1), len(lon1))

    diff = grid1_masked - interp_grid2
    rmse = np.sqrt(np.nanmean(diff ** 2))
    mean = np.nanmean(diff)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(lon_grid, lat_grid, diff, shading='auto', cmap='bwr')
    plt.colorbar(label='Geoid Height Difference (m)')
    plt.title(f'Interpolated Geoid Height Difference (Romania)\nRMSE: {rmse:.4f} m | Mean: {mean:.4f} m')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    if saveas:
        plt.savefig(saveas)
        plt.close()
    else:
        plt.show()

    return rmse, mean

def compare_geoid_heights_interpolated_thresholds(SPG, other_SPG: str, saveas: Optional[str] = None):
    from scipy.interpolate import RegularGridInterpolator
    import geopandas as gpd
    from shapely.geometry import Point
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    # Load Romania polygon
    romania_geojson = Path(__file__).parent / 'romania_polygon.geojson'
    romania_poly = gpd.read_file(romania_geojson).geometry.unary_union

    # Target grid
    meta1 = SPG.data["grids"]["geoid_heights"]["metadata"]
    lat1 = np.linspace(meta1['minphi'], meta1['maxphi'], meta1['nrows'])
    lon1 = np.linspace(meta1['minla'], meta1['maxla'], meta1['ncols'])

    # Source grid
    meta2 = other_SPG.data["grids"]["geoid_heights"]["metadata"]
    lat2 = np.linspace(meta2['minphi'], meta2['maxphi'], meta2['nrows'])
    lon2 = np.linspace(meta2['minla'], meta2['maxla'], meta2['ncols'])
    grid2 = np.squeeze(other_SPG.data["grids"]["geoid_heights"]["grid"])

    interpolator = RegularGridInterpolator((lat2, lon2), grid2, bounds_error=False, fill_value=np.nan)

    # Interpolation points
    lon_grid, lat_grid = np.meshgrid(lon1, lat1)
    flat_points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    mask = np.array([romania_poly.contains(Point(lon, lat)) for lat, lon in flat_points])

    interp_vals = interpolator(flat_points)
    interp_vals[~mask] = np.nan
    interp_grid2 = interp_vals.reshape(len(lat1), len(lon1))

    # Actual grid
    grid1 = np.squeeze(SPG.data["grids"]["geoid_heights"]["grid"])
    diff = grid1 - interp_grid2
    diff[~mask.reshape(len(lat1), len(lon1))] = np.nan

    # Metrics
    abs_diff = np.abs(diff)
    rmse = np.sqrt(np.nanmean(diff**2))
    mean = np.nanmean(diff)

    threshold_counts = {thr: np.nansum(abs_diff < thr) for thr in THRESHOLDS}
    threshold_info = " | ".join([f"<{int(thr*100)}cm: {cnt}" for thr, cnt in threshold_counts.items()])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(lon_grid, lat_grid, diff, shading='auto', cmap='bwr')
    plt.colorbar(label='Geoid Height Difference (m)')
    plt.title(f'Interpolated Geoid Height Difference (Romania)\n'
              f'RMSE: {rmse:.4f} m | Mean: {mean:.4f} m\n{threshold_info}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    if saveas:
        plt.savefig(saveas)
        plt.close()
    else:
        plt.show()

    return rmse, mean, threshold_counts

def compare_geoid_heights_interpolated(SPG, other_SPG: str, saveas: Optional[str] = None):
    from scipy.interpolate import RegularGridInterpolator
    import geopandas as gpd
    from shapely.geometry import Point
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap


    romania_geojson = Path(__file__).parent / 'romania_polygon.geojson'
    romania_poly = gpd.read_file(romania_geojson).geometry.unary_union

    meta1 = SPG.data["grids"]["geoid_heights"]["metadata"]
    lat1 = np.linspace(meta1['minphi'], meta1['maxphi'], meta1['nrows'])
    lon1 = np.linspace(meta1['minla'], meta1['maxla'], meta1['ncols'])

    meta2 = other_SPG.data["grids"]["geoid_heights"]["metadata"]
    lat2 = np.linspace(meta2['minphi'], meta2['maxphi'], meta2['nrows'])
    lon2 = np.linspace(meta2['minla'], meta2['maxla'], meta2['ncols'])
    grid2 = np.squeeze(other_SPG.data["grids"]["geoid_heights"]["grid"])

    interpolator = RegularGridInterpolator((lat2, lon2), grid2, bounds_error=False, fill_value=np.nan)

    lon_grid, lat_grid = np.meshgrid(lon1, lat1)
    flat_points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    mask = np.array([romania_poly.contains(Point(lon, lat)) for lat, lon in flat_points])

    interp_vals = interpolator(flat_points)
    interp_vals[~mask] = np.nan
    interp_grid2 = interp_vals.reshape(len(lat1), len(lon1))

    grid1 = np.squeeze(SPG.data["grids"]["geoid_heights"]["grid"])
    diff = grid1 - interp_grid2
    diff[~mask.reshape(len(lat1), len(lon1))] = np.nan

    abs_diff = np.abs(diff)
    rmse = np.sqrt(np.nanmean(diff**2))
    mean = np.nanmean(diff)

    threshold_counts = {thr: np.nansum(abs_diff < thr) for thr in THRESHOLDS}
    threshold_info = " | ".join([f"<{int(thr*100)}cm: {cnt}" for thr, cnt in threshold_counts.items()])

    # Plot
    plt.figure(figsize=(20, 12))
    cmap = plt.get_cmap("bwr")
    pcm = plt.pcolormesh(lon_grid, lat_grid, diff, shading='auto', cmap=cmap)

    # Custom colormap: white from 0 to 0.12, then fades to red by 1.0
    colors = [(1, 1, 1), (1, 0, 0)]  # white to red
    cmap = LinearSegmentedColormap.from_list("white_to_red_offset", colors, N=256)

    # Apply it with vmin=0, vmax=1 — but stretch white for 0–0.12
    norm = plt.Normalize(vmin=0.12)
    # Use abs_diff for absolute value shading
    pcm = plt.pcolormesh(lon_grid, lat_grid, abs_diff, shading='auto', cmap=cmap, norm=norm)

    # cmap = plt.get_cmap("Reds")
    # pcm = plt.pcolormesh(lon_grid, lat_grid, abs_diff, shading='auto', cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(pcm, label='Geoid Height Difference abs(m)')

    # Contours: detailed fine lines
    cs_detail = plt.contour(lon_grid, lat_grid, abs_diff, levels=THRESHOLDS, colors='black', linewidths=0.05)
    plt.clabel(cs_detail, inline=True, fontsize=7, fmt=lambda val: f"<{int(val * 100)}cm")

    # Contours: grouped bands (thicker) ISOLINES
    band_levels = ISOLINES + [np.nanmax(abs_diff)]
    cs_group = plt.contour(lon_grid, lat_grid, abs_diff, levels=band_levels, colors='green', linewidths=1.0, linestyles='solid')
    plt.clabel(cs_group, inline=True, fontsize=8, fmt=lambda val: f"≤{int(val*100)}cm" if val <= 0.20 else ">20cm")

    plt.title(f'Interpolated Geoid Height Difference (Romania)\n'
              f'{Path(SPG.file_path).name} vs. {Path(other_SPG.file_path).name}\n'
              f'RMSE: {rmse:.4f} m | Mean: {mean:.4f} m\n{threshold_info}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    if saveas:
        plt.savefig(saveas)
        plt.close()
    else:
        plt.show()

    return rmse, mean, threshold_counts


def compare_geoid_heights_offset(SPG, other_SPG, max_offset: int = 3, saveas: Optional[str] = None):
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from pathlib import Path
    import geopandas as gpd
    import numpy as np

    # Load Romania boundary
    romania_geojson = Path(__file__).parent / 'romania_polygon.geojson'
    romania_gdf = gpd.read_file(romania_geojson)
    romania_poly = romania_gdf.geometry.unary_union

    # Load lat/lon grids
    meta1 = SPG.data["grids"]["geoid_heights"]["metadata"]
    meta2 = other_SPG.data["grids"]["geoid_heights"]["metadata"]

    grid1 = np.squeeze(SPG.data["grids"]["geoid_heights"]["grid"])
    grid2 = np.squeeze(other_SPG.data["grids"]["geoid_heights"]["grid"])

    lat1 = np.linspace(meta1['minphi'], meta1['maxphi'], meta1['nrows'])
    lon1 = np.linspace(meta1['minla'], meta1['maxla'], meta1['ncols'])

    # Determine overlap dimensions
    nrows = min(grid1.shape[0], grid2.shape[0])
    ncols = min(grid1.shape[1], grid2.shape[1])
    lat_grid, lon_grid = np.meshgrid(lat1[:nrows], lon1[:ncols], indexing="ij")

    # Create Romania mask
    points = [Point(lon, lat) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    romania_mask = np.array([romania_poly.contains(pt) for pt in points]).reshape(lat_grid.shape)

    # Directions
    directions = {
        'north': (-1, 0), 'south': (1, 0),
        'west': (0, -1), 'east': (0, 1),
        'northwest': (-1, -1), 'northeast': (-1, 1),
        'southwest': (1, -1), 'southeast': (1, 1)
    }

    results = {}

    fig, axes = plt.subplots(
        max_offset, len(directions),
        figsize=(3 * len(directions), 3 * max_offset),
        sharex=True, sharey=True,
        constrained_layout=True  # prevent stretching
    )

    for col, (dir_name, (di, dj)) in enumerate(directions.items()):
        for row, shift in enumerate(range(1, max_offset + 1)):
            i_shift = di * shift
            j_shift = dj * shift

            if i_shift >= 0:
                a = grid1[i_shift:nrows, :]
                b = grid2[:nrows - i_shift, :]
                mask = romania_mask[i_shift:nrows, :]
            else:
                a = grid1[:nrows + i_shift, :]
                b = grid2[-i_shift:nrows, :]
                mask = romania_mask[:nrows + i_shift, :]

            if j_shift >= 0:
                a = a[:, j_shift:ncols]
                b = b[:, :ncols - j_shift]
                mask = mask[:, j_shift:ncols]
            else:
                a = a[:, :ncols + j_shift]
                b = b[:, -j_shift:ncols]
                mask = mask[:, :ncols + j_shift]

            diff = a - b
            diff[~mask] = np.nan
            rmse = np.sqrt(np.nanmean(diff ** 2))
            results[f"{dir_name}_{shift}"] = rmse

            ax = axes[row, col]
            ax.pcolormesh(diff, cmap="bwr", shading="auto")
            ax.set_title(f"{dir_name} {shift} cells\nRMSE={rmse:.4f}")
            ax.set_aspect('equal')
            ax.axis("off")

    # Highlight best RMSE
    best_key = min(results, key=results.get)
    for col, dir_name in enumerate(directions):
        for row, shift in enumerate(range(1, max_offset + 1)):
            if f"{dir_name}_{shift}" == best_key:
                axes[row, col].set_title(f"{best_key}\nBEST RMSE={results[best_key]:.4f}", fontweight="bold", color="green")

    fig.suptitle("Shifted Grid Differences", fontsize=16)
    # plt.tight_layout()

    if saveas:
        plot_path = f"{saveas}"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return results, plot_path
    else:
        plt.show()
        return results

def compare_geoid_heights_offset_by_thresholds(SPG, other_SPG, max_offset: int = 3, thresholds = [0.05, 0.10, 0.15, 0.20], saveas: Optional[str] = None):
    import matplotlib.pyplot as plt
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point
    from pathlib import Path

    # Load Romania polygon
    romania_geojson = Path(__file__).parent / 'romania_polygon.geojson'
    romania_gdf = gpd.read_file(romania_geojson)
    romania_poly = romania_gdf.geometry.unary_union

    # Extract grid metadata and values
    meta1 = SPG.data["grids"]["geoid_heights"]["metadata"]
    meta2 = other_SPG.data["grids"]["geoid_heights"]["metadata"]
    grid1 = np.squeeze(SPG.data["grids"]["geoid_heights"]["grid"])
    grid2 = np.squeeze(other_SPG.data["grids"]["geoid_heights"]["grid"])

    lat1 = np.linspace(meta1['minphi'], meta1['maxphi'], meta1['nrows'])
    lon1 = np.linspace(meta1['minla'], meta1['maxla'], meta1['ncols'])

    nrows = min(grid1.shape[0], grid2.shape[0])
    ncols = min(grid1.shape[1], grid2.shape[1])
    lat_grid, lon_grid = np.meshgrid(lat1[:nrows], lon1[:ncols], indexing="ij")

    # Romania mask
    points = [Point(lon, lat) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    romania_mask = np.array([romania_poly.contains(pt) for pt in points]).reshape(lat_grid.shape)

    directions = {
        'north': (-1, 0), 'south': (1, 0),
        'west': (0, -1), 'east': (0, 1),
        'northwest': (-1, -1), 'northeast': (-1, 1),
        'southwest': (1, -1), 'southeast': (1, 1)
    }


    results = {}

    fig, axes = plt.subplots(max_offset, len(directions), figsize=(3 * len(directions), 3 * max_offset),
                             sharex=True, sharey=True, constrained_layout=True)

    for col, (dir_name, (di, dj)) in enumerate(directions.items()):
        for row, shift in enumerate(range(1, max_offset + 1)):
            i_shift = di * shift
            j_shift = dj * shift

            if i_shift >= 0:
                a = grid1[i_shift:nrows, :]
                b = grid2[:nrows - i_shift, :]
                mask = romania_mask[i_shift:nrows, :]
            else:
                a = grid1[:nrows + i_shift, :]
                b = grid2[-i_shift:nrows, :]
                mask = romania_mask[:nrows + i_shift, :]

            if j_shift >= 0:
                a = a[:, j_shift:ncols]
                b = b[:, :ncols - j_shift]
                mask = mask[:, j_shift:ncols]
            else:
                a = a[:, :ncols + j_shift]
                b = b[:, -j_shift:ncols]
                mask = mask[:, :ncols + j_shift]

            diff = np.abs(a - b)
            diff[~mask] = np.nan

            # Count how many values are under thresholds
            counts = {thr: np.nansum(diff < thr) for thr in thresholds}
            results[f"{dir_name}_{shift}"] = counts

            ax = axes[row - 1, col]
            ax.pcolormesh(diff, cmap="bwr", shading="auto")
            ax.set_aspect('equal')
            ax.axis("off")
            ax.set_title(f"{dir_name} {shift}\\n<5cm={counts[0.05]}")

    # Find best match based on highest <5cm count
    best_key = max(results.items(), key=lambda x: x[1][0.05])[0]
    for col, dir_name in enumerate(directions):
        for row, shift in enumerate(range(1, max_offset + 1)):
            if f"{dir_name}_{shift}" == best_key:
                counts = results[best_key]
                axes[row - 1, col].set_title(
                    f"{best_key}\n<5cm={counts[0.05]}",
                    fontweight="bold", color="green"
                )

    fig.suptitle("Geoid Shift Match by Absolute Difference Thresholds", fontsize=14)

    if saveas:
        path = f"{saveas}"
        plt.savefig(path, dpi=150)
        plt.close()
        return results, path
    else:
        plt.show()
        return results

# endregion

# if __name__ == "__main__":
    
    # spg_file       = ".test/rom_grid3d_04.08.spg"
    # other_spg_file = ".test/rom_grid3d_25.05.spg"

    # spg = SPGFile(spg_file)
    # other_spg = SPGFile(other_spg_file)

    # compare_geoid_heights(spg, other_spg)
    # compare_geodetic_shifts(spg, other_spg, mode='both')
    # compare_geoid_heights_interpolated_nothresholds(spg, other_spg)
    # compare_geoid_heights_interpolated_thresholds(spg, other_spg)
    # compare_geoid_heights_interpolated(spg, other_spg, saveas="408-2505.png")

    # spg_file       = ".test/rom_grid3d_04.08.spg"
    # other_spg_file = ".test/rom_grid3d_25.05_shift.spg"

    # spg = SPGFile(spg_file)
    # other_spg = SPGFile(other_spg_file)
    # compare_geoid_heights_interpolated(spg, other_spg, saveas="408-2505_shift.png")

    # grid_25 = SPGFile('.test/rom_grid3d_25.05.spg')
    # grid_04 = SPGFile('.test/rom_grid3d_04.08.spg')

    # # compare_geoid_heights_shifted(grid_25, grid_04, max_shift=5)

    # # d2 = densify_geoid_grid(grid_25, factor=2.0, method="lanczos")
    # # d2.save_spg('.test/rom_grid3d_25.05_shift_lanczos.spg')

    # compare_geoid_heights_shifted_by_thresholds(grid_04, grid_25, max_shift=2, saveas='shifted_compare_byThreshold_0408_2505_N2.png')
    # compare_geoid_heights_shifted(grid_04, grid_25, max_shift=2, saveas='shifted_compare_RMSE_0408_2505_N2.png')


    # spg_file       = ".test/rom_grid3d_04.08.spg"
    # other_spg_file = ".test/rom_grid3d_25.05_shift_lanczos.spg"

    # spg = SPGFile(spg_file)
    # other_spg = SPGFile(other_spg_file)
    # compare_geoid_heights_interpolated(spg, other_spg, saveas="408-2505_shift_lanczos.png")