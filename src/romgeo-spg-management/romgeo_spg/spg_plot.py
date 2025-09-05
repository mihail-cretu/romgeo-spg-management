import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Literal, List
from pathlib import Path
import json
import itertools
from matplotlib.patches import Polygon as MplPolygon

from .spg_file import SPGFile



def _visualize_heatmap_geodetic(SPG, axis: Literal['x', 'y', 'both'] = None, saveas: Optional[str] = None, use_index: bool = False):
    """Plot a heatmap for the Geodetic Shifts Grid using real-world coordinates."""

    if axis is None:
        axis = 'both'

    min_lon, max_lon = SPG.data["grids"]["geodetic_shifts"]["metadata"]["mine"], SPG.data["grids"]["geodetic_shifts"]["metadata"]["maxe"]  # X-axis (Longitude)
    min_lat, max_lat = SPG.data["grids"]["geodetic_shifts"]["metadata"]["minn"], SPG.data["grids"]["geodetic_shifts"]["metadata"]["maxn"]  # Y-axis (Latitude)

    grid = SPG.data["grids"]["geodetic_shifts"]["grid"]

    if grid.shape[0] != 2:
        raise ValueError("Grid data must be 2D for visualization.")
    
    lon_values = np.linspace(min_lon, max_lon, grid.shape[1])  # Longitude points
    lat_values = np.linspace(min_lat, max_lat, grid.shape[0])  # Latitude points
    
    if axis == 'x':
        flipped_grid = np.flipud(grid[0])

        rmse = np.sqrt(np.nanmean(flipped_grid ** 2))
        title = f'X Shift Difference (RMSE: {rmse:.4f} m)'

    elif axis == 'y':
        flipped_grid = np.flipud(grid[1])

        rmse = np.sqrt(np.nanmean(flipped_grid ** 2))
        title = f'Y Shift Difference (RMSE: {rmse:.4f} m)'

    elif axis == 'both':
        flipped_grid = np.flipud(np.sqrt(grid[0]**2 + grid[1]**2))

        rmse = np.sqrt(np.nanmean(flipped_grid**2))
        title = f'Combined Shift Difference (RMSE: {rmse:.4f} m)'
    else:
        raise ValueError("axis must be 'x', 'y', or 'both'.")
    
    plt.figure(figsize=(10, 6))
    plt.imshow(flipped_grid, cmap="viridis", aspect="auto", interpolation="nearest",
                extent=[min_lon, max_lon, min_lat, max_lat] if not use_index else None)
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("Longitude (°)" if not use_index else "X index")
    plt.ylabel("Latitude (°)" if not use_index else "Y index")

    if saveas:
        plt.savefig(saveas)
        plt.close()
    else:
        plt.show()

def _visualize_heatmap_geoid(SPG:SPGFile, saveas: Optional[str] = None, use_index: bool = False, cmap = "viridis", isolines: Optional[float] = None):
    """Plot a heatmap for the Geoid Heights Grid using real-world coordinates and overlay Romania polygon."""
    min_lon, max_lon = SPG.data["grids"]["geoid_heights"]["metadata"]["minla"],  SPG.data["grids"]["geoid_heights"]["metadata"]["maxla"]  # X-axis (Longitude)
    min_lat, max_lat = SPG.data["grids"]["geoid_heights"]["metadata"]["minphi"], SPG.data["grids"]["geoid_heights"]["metadata"]["maxphi"]  # Y-axis (Latitude)
    
    grid = SPG.data["grids"]["geoid_heights"]["grid"]

    if grid.shape[0] != 1:
        raise ValueError("Grid data must be 1D for visualization.")

    lon_values = np.linspace(min_lon, max_lon, grid.shape[1])  # Longitude points
    lat_values = np.linspace(min_lat, max_lat, grid.shape[2])  # Latitude points
    
    flipped_grid = np.flipud(grid[0])  # Ensure correct vertical alignment
    
    rmse = np.sqrt(np.nanmean(flipped_grid**2))

    plt.figure(figsize=(10, 6))
    plt.imshow(flipped_grid, cmap=cmap, aspect="auto", interpolation="nearest",
                extent=[min_lon, max_lon, min_lat, max_lat] if not use_index else None)
    plt.colorbar(label="Height (m)")
    plt.title(f"Geoid Heights Grid (RMSE: {rmse:.4f} m)")
    plt.xlabel("Longitude (°)" if not use_index else "X index")
    plt.ylabel("Latitude (°)" if not use_index else "Y index")
    plt.grid(True)

    plt.figtext(0.99, 0.01, f"Data source: v{SPG.data['params']['version']}, {Path(SPG.file_path).name}, {SPG.data['grids']['geoid_heights']['name']}", horizontalalignment='right', fontsize=7, color='gray')

    # Add isolines if specified
    if isolines:
        Z =  grid[0] #flipped_grid # original (not flipped)
        # Z = flipped_grid
        lat_values = np.linspace(min_lat, max_lat, Z.shape[0])
        lon_values = np.linspace(min_lon, max_lon, Z.shape[1])
        X, Y = np.meshgrid(lon_values, lat_values)
        levels = np.arange(np.floor(np.nanmin(Z)),
                           np.ceil(np.nanmax(Z)) + 1,
                           isolines)
        cs = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.4)
        plt.clabel(cs, fmt='%d', fontsize=6)

    # Overlay Romania polygon, if not using index coordinates
    if not use_index:
        geojson_path = Path(__file__).parent / 'romania_polygon.geojson'
        if geojson_path.exists():
            with open(geojson_path, 'r', encoding='utf-8') as f:
                geojson = json.load(f)
            for feature in geojson.get('features', []):
                geom = feature.get('geometry', {})
                if geom.get('type') == 'Polygon':
                    for coords in geom.get('coordinates', []):
                        poly = MplPolygon(coords, closed=True, fill=False, edgecolor='black', linewidth=0.7)
                        plt.gca().add_patch(poly)
                elif geom.get('type') == 'MultiPolygon':
                    for polygon in geom.get('coordinates', []):
                        for coords in polygon:
                            poly = MplPolygon(coords, closed=True, fill=False, edgecolor='black', linewidth=0.7)
                            plt.gca().add_patch(poly)

    if saveas:
        plt.savefig(saveas)
        plt.close()
    else:
        plt.show()




def plot_spg_cell(spg_files:List[str], output_path=None, colors=None, labels=None, step:int=10):
    """
    Plot grid overlays for a list of SPG files.

    Args:
        spg_files (list): List of SPG file paths.
        output_path (str or None): If None, show plot. If str, save plot to this path.
        colors (list or None): List of colors for each grid. Defaults to matplotlib cycle.
        labels (list or None): List of labels for each grid. Defaults to file names.
    """
    if not isinstance(spg_files, (list, tuple)):
        spg_files = [spg_files]

    # Load SPG files and extract metadata
    spg_objs = [SPGFile(f) for f in spg_files]
    metas = [spg.get_metadata()["geoid_heights"] for spg in spg_objs]

    # Set colors and labels
    if colors is None:
        colors = list(itertools.islice(plt.rcParams['axes.prop_cycle'].by_key()['color'], len(spg_files)))
    if labels is None:
        labels = [str(f) for f in spg_files]

    fig, ax = plt.subplots(figsize=(12, 9))

    for idx, (meta, color, label) in enumerate(zip(metas, colors, labels)):
        for i in range(0, meta["nrows"], step):
            for j in range(0, meta["ncols"], step):
                lat = meta["minphi"] + i * (meta["stepphi"] )
                lon = meta["minla"]  + j * (meta["stepla"]  )
                ax.add_patch(
                    patches.Rectangle(
                        (lon, lat),
                        meta["stepla"]  * step,
                        meta["stepphi"] * step,
                        linewidth=0.2 if idx else 0.1,
                        edgecolor=color,
                        facecolor='none',
                        label=label if (i == 0 and j == 0) else None
                    )
                )

    # Set plot limits
    min_lon = min(meta["minla"] for meta in metas)
    max_lon = max(meta["maxla"] for meta in metas)
    min_lat = min(meta["minphi"] for meta in metas)
    max_lat = max(meta["maxphi"] for meta in metas)
    ax.set_xlim(min_lon - 0.1, max_lon + 0.1)
    ax.set_ylim(min_lat - 0.1, max_lat + 0.1)

    ax.set_title("Grid Cell Overlay" + f" every {step}x{step} cell" if step > 1 else "")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.legend()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

def visualize_heatmap(SPG, grid_type: Literal["geodetic", "geoid"], saveas: Optional[str] = None, **kwargs):
    """Visualize a heatmap for either geodetic shifts or geoid heights."""
    if grid_type == "geodetic":
        _visualize_heatmap_geodetic(SPG, saveas=saveas, **kwargs)
    elif grid_type == "geoid":
        _visualize_heatmap_geoid(SPG, saveas=saveas, **kwargs)
    else:
        raise ValueError("grid_type must be either 'geodetic' or 'geoid'.")
    
