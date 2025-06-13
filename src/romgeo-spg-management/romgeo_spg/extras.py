from romgeo_spg.spg_file import SPGFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import pickle

def plot_spg_nodes(spg_files, output_path=None, colors=None, labels=None):
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
        for i in range(meta["nrows"]):
            for j in range(meta["ncols"]):
                lat = meta["minphi"] + i * meta["stepphi"]
                lon = meta["minla"] + j * meta["stepla"]
                ax.add_patch(
                    patches.Rectangle(
                        (lon, lat),
                        meta["stepla"],
                        meta["stepphi"],
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

    ax.set_title("Grid Cell Overlay")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True)
    ax.legend()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, dpi=300)
        plt.close(fig)




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

if __name__ == "__main__":
    # Example usage
    output_file = create_snapped_spg_file(
        spg_file="test_rom_grid3d_25.05-beta.spg",
        master_spg_file="test_rom_grid3d_408.spg",
        output_spg_file="rom_grid3d_25.05-snap.spg"
    )
    print(f"Snapped SPG file created: {output_file}")