# test_cubes.py
import numpy as np
import openmc
import openmc.lib
import openmc.mgxs as mgxs
import pytest
from collision_probability import calculate_3d_collision_probability_matrix
from flux import calculate_forward_flux, calculate_adjoint_flux, generate_weight_windows

CUBE_GEOMETRY_CONFIGS = {
    'cube_a': {
        'materials': [
            {'name': 'mat1', 'sigma_total': 1, 'region': 'full'}
        ],
        'boundaries': 'vacuum',
        'size': 2.0,  # 10 cm (cube from -5 to +5 in all axes)
        'mesh_dimension': [3, 3, 3]
    },
    'cube_b': {
        'materials': [
            {'name': 'Sm2O3', 'sigma_total': 211.0, 'region': 'inner', 'size': 4.0},
            {'name': 'H2O', 'sigma_total': 3.45, 'region': 'outer'}
        ],
        'boundaries': 'vacuum',
        'size': 10.0,
        'mesh_dimension': [3, 3, 3]
    },
    'cube_c': {
        'materials': [
            {'name': 'D2O', 'sigma_total': 0.449, 'region': 'diagonal'},
            {'name': 'H2O', 'sigma_total': 3.45, 'region': 'remaining'}
        ],
        'boundaries': 'vacuum',
        'size': 10.0,
        'mesh_dimension': [3, 3, 3]
    },
    'cube_d': {
        'materials': [
            {'name': 'D2O', 'sigma_total': 0.449, 'region': 'dogleg'},
            {'name': 'H2O', 'sigma_total': 3.45, 'region': 'remaining'}
        ],
        'boundaries': 'vacuum',
        'size': 10.0,
        'mesh_dimension': [4, 4, 4]
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
        xs.set_absorption([0.0])
        xs.set_scatter_matrix(np.array([[[0]]]))  # Isotropic scattering
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
        elif config_name == 'cube_b':
            pass
        elif config_name == 'cube_c':
            pass
        elif config_name == 'cube_d':
            pass

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
    
    config = CUBE_GEOMETRY_CONFIGS[config_name]
    num_voxels = np.prod(mesh.dimension)
    print(f'num voxels: {num_voxels}')

    sigma_value = config['materials'][0]['sigma_total']
    sigmas = np.full(num_voxels, sigma_value)
    print(f"sigmas: {sigmas}")

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

    P = calculate_3d_collision_probability_matrix(mesh, volumes, sigmas, num_rays=500)
    print("Collision Probability Matrix:")
    print(P)

    # validation checks
    assert np.all(P >= 0) and np.all(P <= 1), "Invalid probability values"
    assert np.allclose(P.sum(axis=1), 1.0, rtol=0.1), "Conservation violation"
    sigma_V = np.diag(sigmas * volumes)
    assert np.allclose(sigma_V @ P, (sigma_V @ P).T, rtol=0.1), "Reciprocity violation"

    # Flux calculations
    forward_flux = calculate_forward_flux(P, sigmas, volumes, forward_source, method='direct', max_iter=10000, tol=1e-6)
    print("\nForward Flux:")
    print(forward_flux)
    adjoint_flux = calculate_adjoint_flux(P, sigmas, volumes, adjoint_source, method='direct', max_iter=10000, tol=1e-6)
    print("\nAdjoint Flux:")
    print(adjoint_flux)

    # Weight window generation
    w_lower, w_upper = generate_weight_windows(adjoint_flux, alpha=1.0, C_w=5.0)

    assert np.all(w_lower > 0), "Invalid lower weight bounds"
    assert np.all(w_upper > w_lower), "Invalid upper weight bounds"

    openmc.lib.finalize()

if __name__ == "__main__":
    for config in ['cube_a']:
        test_cube_collision_probability(config)