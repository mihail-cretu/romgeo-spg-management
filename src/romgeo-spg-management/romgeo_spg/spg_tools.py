from scipy.ndimage import zoom
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal
import copy
import pickle

from .spg_file import SPGFile



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

def create_snapped_spg_file(spg_file, master_spg_file, output_spg_file):
    """
    Create a new SPG file by snapping the beta grid to the nearest base grid cell.
    Args:
        spg_file (str): Path to the SPG file containing the beta grid.
        master_spg_file (str): Path to the SPG file containing the base grid.
        output_spg_file (str): Path to save the new snapped SPG file.
    Returns:
        str: Path to the newly created snapped SPG file.
    """
    # Load both SPG files
    SPG = SPGFile(spg_file)
    master_SPG = SPGFile(master_spg_file)

    # Grid metadata
    grid_2505_meta = SPG.data["grids"]["geoid_heights"]["metadata"]
    grid_408_meta = master_SPG.data["grids"]["geoid_heights"]["metadata"]

    # Grid step
    step_lat = grid_408_meta["stepphi"]
    step_lon = grid_408_meta["stepla"]

    # Snap beta min coords to nearest base cell
    snapped_lat = grid_408_meta["minphi"] + round((grid_2505_meta["minphi"] - grid_408_meta["minphi"]) / step_lat) * step_lat
    snapped_lon = grid_408_meta["minla"] + round((grid_2505_meta["minla"] - grid_408_meta["minla"]) / step_lon) * step_lon

    # Index offset in base grid
    lat_idx = round((snapped_lat - grid_408_meta["minphi"]) / step_lat)
    lon_idx = round((snapped_lon - grid_408_meta["minla"]) / step_lon)

    # Size of beta grid
    nrows = grid_2505_meta["nrows"]
    ncols = grid_2505_meta["ncols"]

    # Extract subgrid from base
    new_geoid = master_SPG.data["grids"]["geoid_heights"]["grid"][0][lat_idx:lat_idx+nrows, lon_idx:lon_idx+ncols]

    # Update beta data in-place
    SPG.data["grids"]["geoid_heights"]["grid"][0] = new_geoid
    SPG.data["grids"]["geoid_heights"]["metadata"].update({
        "minphi": snapped_lat,
        "maxphi": snapped_lat + (nrows - 1) * step_lat,
        "minla": snapped_lon,
        "maxla": snapped_lon + (ncols - 1) * step_lon,
        "nrows": nrows,
        "ncols": ncols
    })

    # Save to new file
    with open(output_spg_file, "wb") as f:
        pickle.dump(SPG.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return output_spg_file

    
# Example usage
# output_file = create_snapped_spg_file(
#     spg_file="test_rom_grid3d_25.05-beta.spg",
#     master_spg_file="test_rom_grid3d_408.spg",
#     output_spg_file="rom_grid3d_25.05-snap.spg"
# )
# print(f"Snapped SPG file created: {output_file}")