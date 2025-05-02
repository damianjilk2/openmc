# test_cubes.py
import numpy as np
import openmc
import openmc.lib
import openmc.mgxs as mgxs
import pytest
from collision_probability import calculate_3d_collision_probability_matrix
from flux import calculate_forward_flux, calculate_adjoint_flux, generate_weight_windows
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import seaborn as sns

CUBE_GEOMETRY_CONFIGS = {
    'cube_a': {
        'materials': [
            {'name': 'mat1', 'sigma_total': 1, 'sigma_scattering': 0.5, 'region': 'full'}
        ],
        'boundaries': 'vacuum',
        'size': 10.0,  # 10 cm (cube from -5 to +5 in all axes)
        'mesh_dimension': [3, 3, 3]
    }
}

def create_cube_model(config_name: str):
    """Create 3D cube geometry with specified material configuration."""
    config = CUBE_GEOMETRY_CONFIGS[config_name]
    size = config['size']
    half_size = size / 2

    x_min = openmc.XPlane(-half_size, boundary_type=config['boundaries'])
    x_max = openmc.XPlane(half_size, boundary_type=config['boundaries'])
    y_min = openmc.YPlane(-half_size, boundary_type=config['boundaries'])
    y_max = openmc.YPlane(half_size, boundary_type=config['boundaries'])
    z_min = openmc.ZPlane(-half_size, boundary_type=config['boundaries'])
    z_max = openmc.ZPlane(half_size, boundary_type=config['boundaries'])

    materials = []
    groups = mgxs.EnergyGroups([1e-5, 1e6])
    mgxs_lib = openmc.MGXSLibrary(groups)
    
    for mat_config in config['materials']:
        xs = openmc.XSdata(mat_config['name'], groups)
        xs.order = 0
        xs.set_total([mat_config['sigma_total']])
        xs.set_absorption([mat_config['sigma_total'] - mat_config['sigma_scattering']])
        xs.set_scatter_matrix(np.array([[[mat_config['sigma_scattering']]]]))
        mgxs_lib.add_xsdata(xs)
        
        mat = openmc.Material(name=mat_config['name'])
        mat.add_macroscopic(mat_config['name'])
        materials.append(mat)

    mgxs_lib.export_to_hdf5('cube_xs.h5')
    materials = openmc.Materials(materials)
    materials.cross_sections = 'cube_xs.h5'
    materials.export_to_xml()

    cells = []

    for mat_config in config['materials']:
        if config_name == 'cube_a':
            region = -x_max & +x_min & -y_max & +y_min & -z_max & +z_min
        # elif config_name == 'cube_b':
        #     pass
        # elif config_name == 'cube_c':
        #     pass
        # elif config_name == 'cube_d':
        #     pass

        cell = openmc.Cell(region=region, fill=mat)
        cells.append(cell)

    geometry = openmc.Geometry(cells)
    geometry.export_to_xml()

    settings = openmc.Settings()
    settings.energy_mode = "multi-group"
    settings.particles = 1
    settings.batches = 1
    settings.export_to_xml()

    mesh = openmc.RegularMesh()
    mesh.dimension = config['mesh_dimension']
    mesh.lower_left = [-half_size]*3
    mesh.upper_right = [half_size]*3

    return mesh

@pytest.mark.parametrize("config_name", ['cube_a'])
def test_cube_collision_probability(config_name):
    """Test collision probability calculation for 3D cube configurations."""
    mesh = create_cube_model(config_name)
    openmc.lib.init()
    
    num_voxels = np.prod(mesh.dimension)
    print(f'num voxels: {num_voxels}')

    upper_right = np.array(mesh.upper_right)
    lower_left = np.array(mesh.lower_left)
    total_volume = np.prod(upper_right - lower_left)
    voxel_volume = total_volume / num_voxels
    volumes = np.full(num_voxels, voxel_volume)
    print(f"volumes: {volumes}")

    forward_source = np.zeros(num_voxels)
    forward_source[num_voxels//2] = 1.0
    adjoint_source = np.zeros(num_voxels)
    adjoint_source[num_voxels//2] = 1.0

    P, sigma_t, sigma_s = calculate_3d_collision_probability_matrix(mesh, num_rays=10000)
    print("Collision Probability Matrix:")
    print(P)
    print("Total Cross Section:")
    print(sigma_t)
    print("Scattering Cross Section:")
    print(sigma_s)

    invalid_mask = (P < 0) | (P > 1)
    if np.any(invalid_mask):
        invalid_indices = np.argwhere(invalid_mask)
        for i, j in invalid_indices:
            print(f"Invalid P[{i}, {j}] = {P[i, j]}")
        raise ValueError("P matrix contains invalid probability values (not in [0, 1])")

    forward_flux = calculate_forward_flux(P, sigma_t, sigma_s, volumes, forward_source, method='direct')
    print("\nForward Flux:")
    print(forward_flux)

    adjoint_flux = calculate_adjoint_flux(P, sigma_t, sigma_s, volumes, adjoint_source, method='direct')
    print("\nAdjoint Flux:")
    print(adjoint_flux)

    plot_probability_matrix(P, mesh.dimension, flux=forward_flux, adjoint_flux=adjoint_flux)

    openmc.lib.finalize()

def plot_probability_matrix(P, mesh_dimensions, flux=None, adjoint_flux=None,
                            flux_label='Forward Flux', adjoint_flux_label='Adjoint Flux',
                            filename='collision_probability_matrix.png'):
    """
    Plot and save collision probability matrix with optional flux and adjoint flux overlays.
    
    Parameters:
    - P (ndarray): NxN collision probability matrix
    - mesh_dimensions (tuple): (Nx, Ny, Nz) voxel mesh dimensions
    - flux (ndarray or None): Optional length-N forward flux vector
    - adjoint_flux (ndarray or None): Optional length-N adjoint flux vector
    - flux_label (str): Label for the forward flux axis
    - adjoint_flux_label (str): Label for the adjoint flux axis
    - filename (str): Output filename
    """
    N = P.shape[0]

    side_plots = sum(x is not None for x in [flux, adjoint_flux])
    fig = plt.figure(figsize=(14 + 2 * side_plots, 10))
    spec = gridspec.GridSpec(ncols=1 + side_plots, nrows=1, width_ratios=[5] + [1]*side_plots)

    ax0 = fig.add_subplot(spec[0])
    sns.heatmap(
        P,
        norm=LogNorm(vmin=P.min(), vmax=P.max()),
        cmap='viridis',
        ax=ax0,
        cbar_kws={'label': 'Collision Probability'},
        square=True
    )
    ax0.set_xlabel('Destination Voxel (j)')
    ax0.set_ylabel('Source Voxel (i)')
    ax0.set_title(f'Collision Probability Matrix ({mesh_dimensions[0]}x{mesh_dimensions[1]}x{mesh_dimensions[2]})')

    # forward and adjoint fluxes if provided
    side_index = 1
    if flux is not None:
        ax_flux = fig.add_subplot(spec[side_index], sharey=ax0)
        ax_flux.plot(flux, np.arange(N), marker='o', linestyle='-', color='orange')
        ax_flux.set_title(flux_label)
        ax_flux.set_xlabel(flux_label)
        ax_flux.grid(True)
        plt.setp(ax_flux.get_yticklabels(), visible=False)
        side_index += 1

    if adjoint_flux is not None:
        ax_adj = fig.add_subplot(spec[side_index], sharey=ax0)
        ax_adj.plot(adjoint_flux, np.arange(N), marker='x', linestyle='-', color='blue')
        ax_adj.set_title(adjoint_flux_label)
        ax_adj.set_xlabel(adjoint_flux_label)
        ax_adj.grid(True)
        plt.setp(ax_adj.get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    for config in ['cube_a']:
        test_cube_collision_probability(config)