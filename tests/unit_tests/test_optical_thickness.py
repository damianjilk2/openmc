import openmc
from openmc import RegularMesh
import openmc.lib
import numpy as np
import openmc.mgxs as mgxs
import pytest

@pytest.fixture
def my_model():
    groups = mgxs.EnergyGroups(group_edges=[1e-5, 1.0e6])

    scatter_matrix = np.array([[[0]]])

    mat1_xsdata = openmc.XSdata('mat1', groups)
    mat1_xsdata.order = 0
    mat1_xsdata.set_total([0.01])
    mat1_xsdata.set_absorption([0])
    mat1_xsdata.set_scatter_matrix(scatter_matrix)

    mat2_xsdata = openmc.XSdata('mat2', groups)
    mat2_xsdata.order = 0
    mat2_xsdata.set_total([0.02])
    mat2_xsdata.set_absorption([0])
    mat2_xsdata.set_scatter_matrix(scatter_matrix)

    one_g_XS_file = openmc.MGXSLibrary(groups)
    one_g_XS_file.add_xsdatas([mat1_xsdata, mat2_xsdata])
    one_g_XS_file.export_to_hdf5('xs.h5')


    # Create a dummy material with fictitious cross-section data
    material1 = openmc.Material(name="Material1")
    material1.add_macroscopic('mat1')

    material2 = openmc.Material(name="Material2")
    material2.add_macroscopic('mat2')

    materials = openmc.Materials([material1, material2])
    materials.cross_sections = 'xs.h5'
    materials.export_to_xml()

    sphere_inner = openmc.Sphere(r=2, boundary_type='transmission')
    sphere_outer = openmc.Sphere(r=10, boundary_type='vacuum')

    # Inner cell (filled with Material1)
    cell_inner = openmc.Cell(name="Inner Sphere", fill=material1, region=-sphere_inner)

    # Outer shell (filled with Material2)
    cell_outer = openmc.Cell(name="Outer Sphere", fill=material2, region=+sphere_inner & -sphere_outer)

    geometry = openmc.Geometry([cell_inner, cell_outer])
    geometry.export_to_xml()


    settings = openmc.Settings()
    settings.energy_mode = "multi-group"
    settings.particles = 1
    settings.batches = 1
    settings.export_to_xml()

@pytest.fixture
def lib_init(my_model):
    openmc.lib.init()
    yield
    openmc.lib.finalize()

@pytest.fixture
def lib_simulation_init(lib_init):
    openmc.lib.simulation_init()
    yield

def test_optical_thickness_variations(lib_init):
    """Test optical thickness calculation for various scenarios."""
    mat1_xs = 0.01
    mat2_xs = 0.02
    mat1_r = 2
    mat2_r = 10

    # Origin to outer boundary in x-direction
    expected_tau_x = mat1_xs * mat1_r + mat2_xs * (mat2_r - mat1_r)
    calculated_tau_x = openmc.lib.calculate_optical_thickness((0, 0, 0), (10, 0, 0))
    assert np.isclose(calculated_tau_x, expected_tau_x)

    # Origin to boundary between regions in x-direction
    expected_tau_boundary = mat1_xs * mat1_r
    calculated_tau_boundary = openmc.lib.calculate_optical_thickness((0, 0, 0), (2, 0, 0))
    assert np.isclose(calculated_tau_boundary, expected_tau_boundary)

    # Origin to diagonal outer boundary
    distance_diag = 10
    expected_tau_diag = mat1_xs * mat1_r + mat2_xs * (distance_diag - mat1_r)
    calculated_tau_diag = openmc.lib.calculate_optical_thickness((0, 0, 0), (5.8, 5.8, 5.8))
    assert np.isclose(calculated_tau_diag, expected_tau_diag)

def test_voxel_optical_thickness(lib_init):
    """Test voxel-based optical thickness calculations."""
    mesh = RegularMesh()
    mesh.dimension = (2, 2, 2)
    mesh.lower_left = (0, 0, 0)
    mesh.upper_right = (10, 10, 10)

    num_rays = 100
    tau = calculate_optical_thickness_for_voxels(mesh, num_rays)

    assert tau.shape == (8, 8)
    assert np.all(tau >= 0)

def calculate_optical_thickness_for_voxels(mesh: RegularMesh, num_rays: int):
    lower_left = np.array(mesh.lower_left)
    upper_right = np.array(mesh.upper_right)
    dimensions = np.array(mesh.dimension)

    # Calculate voxel size in each dimension
    voxel_size = (upper_right - lower_left) / dimensions

    # Preallocate optical thickness matrix
    num_voxels = np.prod(dimensions)
    tau = np.zeros((num_voxels, num_voxels))

    # Helper function to compute voxel bounds given its index
    def voxel_bounds(index):
        i = index % dimensions[0]
        j = (index // dimensions[0]) % dimensions[1]
        k = index // (dimensions[0] * dimensions[1])
        min_bound = lower_left + voxel_size * np.array([i, j, k])
        max_bound = min_bound + voxel_size
        return min_bound, max_bound

    # Loop over voxel pairs
    for start_voxel in range(num_voxels):
        start_min, start_max = voxel_bounds(start_voxel)
        for end_voxel in range(num_voxels):
            end_min, end_max = voxel_bounds(end_voxel)
            tau[start_voxel, end_voxel] = openmc.lib.get_mean_optical_thickness_between_voxels(
                start_min, start_max, end_min, end_max, num_rays
            )
    return tau