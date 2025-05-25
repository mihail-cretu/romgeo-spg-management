import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal

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
        flipped_grid = np.flipud(flipped_grid[1])

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

def _visualize_heatmap_geoid(SPG, saveas: Optional[str] = None, use_index: bool = False):
    """Plot a heatmap for the Geoid Heights Grid using real-world coordinates."""
    min_lon, max_lon = SPG.data["grids"]["geoid_heights"]["metadata"]["minla"],  SPG.data["grids"]["geoid_heights"]["metadata"]["maxla"]  # X-axis (Longitude)
    min_lat, max_lat = SPG.data["grids"]["geoid_heights"]["metadata"]["minphi"], SPG.data["grids"]["geoid_heights"]["metadata"]["maxphi"]  # Y-axis (Latitude)
    
    grid = SPG.data["grids"]["geoid_heights"]["grid"]

    if grid.shape[0] != 1:
        raise ValueError("Grid data must be 1D for visualization.")

    lon_values = np.linspace(min_lon, max_lon, grid.shape[1])  # Longitude points
    lat_values = np.linspace(min_lat, max_lat, grid.shape[0])  # Latitude points
    
    flipped_grid = np.flipud(grid[0])  # Ensure correct vertical alignment
    
    rmse = np.sqrt(np.nanmean(flipped_grid**2))

    plt.figure(figsize=(10, 6))
    plt.imshow(flipped_grid, cmap="viridis", aspect="auto", interpolation="nearest",
                extent=[min_lon, max_lon, min_lat, max_lat] if not use_index else None)
    plt.colorbar(label="Value")
    plt.title(f"Geoid Heights Grid (RMSE: {rmse:.4f} m)")
    plt.xlabel("Longitude (°)" if not use_index else "X index")
    plt.ylabel("Latitude (°)" if not use_index else "Y index")

    if saveas:
        plt.savefig(saveas)
        plt.close()
    else:
        plt.show()

def visualize_heatmap(SPG, grid_type: Literal["geodetic", "geoid"], saveas: Optional[str] = None, **kwargs):
    """Visualize a heatmap for either geodetic shifts or geoid heights."""
    if grid_type == "geodetic":
        _visualize_heatmap_geodetic(SPG, saveas=saveas, **kwargs)
    elif grid_type == "geoid":
        _visualize_heatmap_geoid(SPG, saveas=saveas, **kwargs)
    else:
        raise ValueError("grid_type must be either 'geodetic' or 'geoid'.")
    
