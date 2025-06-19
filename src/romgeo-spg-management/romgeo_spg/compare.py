import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal
import json
from pathlib import Path

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
# THRESHOLDS = [0.06, 0.12, 0.3]
ISOLINES = [0.12]  # 12cm and 20cm bands

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


if __name__ == "__main__":
    from spg_file import SPGFile

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




    spg_file       = ".test/rom_grid3d_04.08.spg"
    other_spg_file = ".test/rom_grid3d_25.05_shift_lanczos.spg"

    spg = SPGFile(spg_file)
    other_spg = SPGFile(other_spg_file)
    compare_geoid_heights_interpolated(spg, other_spg, saveas="408-2505_shift_lanczos.png")