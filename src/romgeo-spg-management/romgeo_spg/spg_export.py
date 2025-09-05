
import rasterio
import numpy as np
from rasterio.transform import from_bounds
from scipy.ndimage import zoom, map_coordinates
from scipy.interpolate import interp2d, RectBivariateSpline

from .spg_file import SPGFile

def _interpolate_grid(data, scale, method):
    if scale == 1.0 and method == "none":
        return data

    if method == "none":
        # Just resize using numpy repeat (nearest neighbor)
        return np.repeat(np.repeat(data, int(scale), axis=0), int(scale), axis=1)
    elif method == "nearest":
        return zoom(data, scale, order=0)
    elif method == "linear":
        return zoom(data, scale, order=1)
    elif method == "cubic":
        return zoom(data, scale, order=3)
    elif method == "lanczos":
        # Lanczos is not directly available; use order=4 as an approximation
        return zoom(data, scale, order=4)
    elif method == "spline":
        # Use RectBivariateSpline for smooth spline interpolation
        y = np.arange(data.shape[0])
        x = np.arange(data.shape[1])
        spline = RectBivariateSpline(y, x, data)
        y_new = np.linspace(0, data.shape[0] - 1, int(data.shape[0] * scale))
        x_new = np.linspace(0, data.shape[1] - 1, int(data.shape[1] * scale))
        return spline(y_new, x_new)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

def save_geoid_as_geotiff(SPG:SPGFile, filename: str, interpolation: str = "none", scale: float = 1.0):
    """
    Save the geoid heights grid as an uncompressed 64-bit GeoTIFF with georeferencing.
    interpolation: one of 'none', 'nearest', 'linear', 'cubic', 'lanczos', 'spline'
    scale: scaling factor for upsampling/downsampling (default 1.0)
    """
    min_lon = SPG.data["grids"]["geoid_heights"]["metadata"]["minla"]
    max_lon = SPG.data["grids"]["geoid_heights"]["metadata"]["maxla"]
    min_lat = SPG.data["grids"]["geoid_heights"]["metadata"]["minphi"]
    max_lat = SPG.data["grids"]["geoid_heights"]["metadata"]["maxphi"]
    grid    = SPG.data["grids"]["geoid_heights"]["grid"]

    if grid.shape[0] != 1:
        raise ValueError("Grid data must be 1D for saving as GeoTIFF.")

    data = np.flipud(grid[0])
    data = _interpolate_grid(data, scale, interpolation)

    height, width = data.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float64',
        crs='EPSG:4326',
        transform=transform,
        compress='none'
    ) as dst:
        dst.write(data, 1)

def save_geodetic_shifts_as_geotiff(SPG: SPGFile, filename: str, interpolation: str = "none", scale: float = 1.0):
    """
    Save the geodetic shifts grid (2-band: East, North) as an uncompressed 64-bit GeoTIFF with georeferencing.
    interpolation: one of 'none', 'nearest', 'linear', 'cubic', 'lanczos', 'spline'
    scale: scaling factor for upsampling/downsampling (default 1.0)
    """
    min_e = SPG.data["grids"]["geodetic_shifts"]["metadata"]["mine"]
    max_e = SPG.data["grids"]["geodetic_shifts"]["metadata"]["maxe"]
    min_n = SPG.data["grids"]["geodetic_shifts"]["metadata"]["minn"]
    max_n = SPG.data["grids"]["geodetic_shifts"]["metadata"]["maxn"]
    grid  = SPG.data["grids"]["geodetic_shifts"]["grid"]

    if grid.shape[0] != 2:
        raise ValueError("Geodetic shifts grid must have 2 bands (East, North).")

    east_data = np.flipud(grid[0])
    north_data = np.flipud(grid[1])

    east_data = _interpolate_grid(east_data, scale, interpolation)
    north_data = _interpolate_grid(north_data, scale, interpolation)

    height, width = east_data.shape
    transform = from_bounds(min_e, min_n, max_e, max_n, width, height)

    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=2,
        dtype='float64',
        crs='EPSG:3844',
        transform=transform,
        compress='none'
    ) as dst:
        dst.write(east_data, 1)
        dst.set_band_description(1, "East shift")
        dst.write(north_data, 2)
        dst.set_band_description(2, "North shift")

# Example usage:
# spg_file = SPGFile('.test/rom_grid3d_25.05.spg')
# save_geoid_as_geotiff(spg_file, 'rom_grid3d_25.05_geoid_cubic_2.tif', interpolation='cubic', scale=2.0)
# save_geoid_as_geotiff(spg_file, 'rom_grid3d_25.05_geoid_none_1.tif', interpolation='none', scale=1.0)
# save_geodetic_shifts_as_geotiff(spg_file, 'rom_grid3d_25.05_geodetic_shifts_cubic_2.tif', interpolation='cubic', scale=2.0)