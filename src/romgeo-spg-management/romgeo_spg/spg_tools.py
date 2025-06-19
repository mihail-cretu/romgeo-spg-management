from scipy.ndimage import zoom
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal
import copy

from spg_file import SPGFile


def compare_geoid_heights_shifted(SPG, other_SPG, max_shift: int = 3, saveas: Optional[str] = None):
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
        max_shift, len(directions),
        figsize=(3 * len(directions), 3 * max_shift),
        sharex=True, sharey=True,
        constrained_layout=True  # prevent stretching
    )

    for col, (dir_name, (di, dj)) in enumerate(directions.items()):
        for row, shift in enumerate(range(1, max_shift + 1)):
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
        for row, shift in enumerate(range(1, max_shift + 1)):
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


def compare_geoid_heights_shifted_by_thresholds(SPG, other_SPG, max_shift: int = 3, thresholds = [0.05, 0.10, 0.15, 0.20], saveas: Optional[str] = None):
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

    fig, axes = plt.subplots(max_shift, len(directions), figsize=(3 * len(directions), 3 * max_shift),
                             sharex=True, sharey=True, constrained_layout=True)

    for col, (dir_name, (di, dj)) in enumerate(directions.items()):
        for row, shift in enumerate(range(1, max_shift + 1)):
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
        for row, shift in enumerate(range(1, max_shift + 1)):
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


def densify_geoid_grid(spg: SPGFile, factor: float = 2.0, method: str = "cubic") -> SPGFile:
    """
    Densifies the geoid_heights grid in the SPGFile using specified interpolation.
    Returns a new SPGFile with updated grid and metadata.

    Parameters:
        spg (SPGFile): Original SPGFile object to densify.
        zoom_factor (float): Densification factor (default: 2.0).
        method (str): Interpolation method: 'nearest', 'linear', 'cubic', 'lanczos'

    Returns:
        SPGFile: New SPGFile object with densified grid and updated metadata.
    """
    method_map = {
        "nearest": 0,
        "linear": 1,
        "cubic": 3,
        "lanczos": 5
    }

    if method not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from: {list(method_map)}")

    order = method_map[method]
    spg_new = copy.deepcopy(spg)

    grid = np.squeeze(spg_new.data["grids"]["geoid_heights"]["grid"])
    meta = spg_new.data["grids"]["geoid_heights"]["metadata"]

    # Interpolate
    dense_grid = zoom(grid, zoom=factor, order=order)

    # Update metadata
    nrows_new, ncols_new = dense_grid.shape
    stepphi_new = meta["stepphi"] / factor
    stepla_new = meta["stepla"] / factor
    maxphi_new = meta["minphi"] + (nrows_new - 1) * stepphi_new
    maxla_new = meta["minla"] + (ncols_new - 1) * stepla_new

    meta.update({
        "nrows": nrows_new,
        "ncols": ncols_new,
        "stepphi": stepphi_new,
        "stepla": stepla_new,
        "maxphi": maxphi_new,
        "maxla": maxla_new,
    })

    spg_new.data["grids"]["geoid_heights"]["grid"] = dense_grid[np.newaxis, :, :]
    return spg_new


if __name__ == "__main__": 
    
    grid_25 = SPGFile('.test/rom_grid3d_25.05.spg')
    grid_04 = SPGFile('.test/rom_grid3d_04.08.spg')

    # compare_geoid_heights_shifted(grid_25, grid_04, max_shift=5)

    # d2 = densify_geoid_grid(grid_25, factor=2.0, method="lanczos")
    # d2.save_spg('.test/rom_grid3d_25.05_shift_lanczos.spg')

    compare_geoid_heights_shifted_by_thresholds(grid_04, grid_25, max_shift=2, saveas='shifted_compare_byThreshold_0408_2505_N2.png')
    compare_geoid_heights_shifted(grid_04, grid_25, max_shift=2, saveas='shifted_compare_RMSE_0408_2505_N2.png')
    
