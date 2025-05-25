import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal



def compare_geoid_heights(SPG, other_SPG: str, saveas: Optional[str] = None):
    
    lat1 = np.linspace(SPG.data["grids"]["geoid_heights"]["metadata"]['minphi'], SPG.data["grids"]["geoid_heights"]["metadata"]['maxphi'], SPG.data["grids"]["geoid_heights"]["metadata"]['nrows'])
    lon1 = np.linspace(SPG.data["grids"]["geoid_heights"]["metadata"]['minla'],  SPG.data["grids"]["geoid_heights"]["metadata"]['maxla'],  SPG.data["grids"]["geoid_heights"]["metadata"]['ncols'])

    lat2 = np.linspace(other_SPG.geoid_metadata['minphi'], other_SPG.geoid_metadata['maxphi'], other_SPG.geoid_metadata['nrows'])
    lon2 = np.linspace(other_SPG.geoid_metadata['minla'],  other_SPG.geoid_metadata['maxla'],  other_SPG.geoid_metadata['ncols'])

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
    sub2 = np.squeeze(other_SPG.geoid_heights)[np.ix_(idx_lat2, idx_lon2)]

    diff = sub1 - sub2
    rmse = np.sqrt(np.nanmean(diff ** 2))
    mean = np.nanmean(diff)

    lon_grid, lat_grid = np.meshgrid(lon1[idx_lon1], lat1[idx_lat1])

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(lon_grid, lat_grid, diff, shading='auto', cmap='bwr')
    plt.colorbar(label='Geoid Height Difference (m)')
    plt.title(f'Geoid Height Difference\nRMSE: {rmse:.4f} m | Mean: {mean:.4f} m')
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
    n2 = np.linspace(other_SPG.geodetic_metadata['minn'], other_SPG.geodetic_metadata['maxn'], other_SPG.geodetic_metadata['nrows'])
    e2 = np.linspace(other_SPG.geodetic_metadata['mine'], other_SPG.geodetic_metadata['maxe'], other_SPG.geodetic_metadata['ncols'])

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

    sub1 = SPG.data["grids"]["geodetic_shifts"]["grid"][:, idx_n1[:, None], idx_e1]
    sub2 = other_SPG.geodetic_shifts[:, idx_n2[:, None], idx_e2]

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

def compare_geoid_heights_interpolated(SPG, other_SPG: str, saveas: Optional[str] = None):
    from scipy.interpolate import RegularGridInterpolator

    # Target grid (SPG)
    lat1 = np.linspace(SPG.data["grids"]["geoid_heights"]["metadata"]['minphi'], SPG.data["grids"]["geoid_heights"]["metadata"]['maxphi'], SPG.data["grids"]["geoid_heights"]["metadata"]['nrows'])
    lon1 = np.linspace(SPG.data["grids"]["geoid_heights"]["metadata"]['minla'], SPG.data["grids"]["geoid_heights"]["metadata"]['maxla'], SPG.data["grids"]["geoid_heights"]["metadata"]['ncols'])

    # Source grid (other)
    lat2 = np.linspace(other_SPG.geoid_metadata['minphi'], other_SPG.geoid_metadata['maxphi'], other_SPG.geoid_metadata['nrows'])
    lon2 = np.linspace(other_SPG.geoid_metadata['minla'], other_SPG.geoid_metadata['maxla'], other_SPG.geoid_metadata['ncols'])

    grid2 = np.squeeze(other_SPG.geoid_heights)
    interpolator = RegularGridInterpolator((lat2, lon2), grid2, bounds_error=False, fill_value=np.nan)

    # Generate coordinate pairs for interpolation
    lon_grid, lat_grid = np.meshgrid(lon1, lat1)
    points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    interp_grid2 = interpolator(points).reshape(len(lat1), len(lon1))

    # Compute difference
    grid1 = np.squeeze(SPG.data["grids"]["geoid_heights"]["grid"])
    diff = grid1 - interp_grid2
    rmse = np.sqrt(np.nanmean(diff ** 2))
    mean = np.nanmean(diff)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(lon_grid, lat_grid, diff, shading='auto', cmap='bwr')
    plt.colorbar(label='Geoid Height Difference (m)')
    plt.title(f'Interpolated Geoid Height Difference\nRMSE: {rmse:.4f} m | Mean: {mean:.4f} m')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    if saveas:
        plt.savefig(saveas)
        plt.close()
    else:
        plt.show()

    return rmse, mean