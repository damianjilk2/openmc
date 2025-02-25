import openmc
from openmc import RegularMesh
import openmc.lib
import numpy as np
import openmc.mgxs as mgxs
import pytest

@pytest.fixture
def setup_vars():
    return {
        "mat1_xs": 0.01,
        "mat2_xs": 0.02,
        "outer_xs": 0.03,
        "mat1_r": 2,
        "mat2_r": 2,
        "outer_r": 10,
        "left_cell_offset": 3,
        "right_cell_offset": 3
    }

@pytest.fixture
def my_model(setup_vars):
    mat1_xs = setup_vars["mat1_xs"]
    mat2_xs = setup_vars["mat2_xs"]
    outer_xs = setup_vars["outer_xs"]
    mat1_r = setup_vars["mat1_r"]
    mat2_r = setup_vars["mat2_r"]
    outer_r = setup_vars["outer_r"]
    left_cell_offset = setup_vars["left_cell_offset"]
    right_cell_offset = setup_vars["right_cell_offset"]

    groups = mgxs.EnergyGroups(group_edges=[1e-5, 1.0e6])

    scatter_matrix = np.array([[[0]]])

    mat1_xsdata = openmc.XSdata('mat1', groups)
    mat1_xsdata.order = 0
    mat1_xsdata.set_total([mat1_xs])
    mat1_xsdata.set_absorption([0])
    mat1_xsdata.set_scatter_matrix(scatter_matrix)

    mat2_xsdata = openmc.XSdata('mat2', groups)
    mat2_xsdata.order = 0
    mat2_xsdata.set_total([mat2_xs])
    mat2_xsdata.set_absorption([0])
    mat2_xsdata.set_scatter_matrix(scatter_matrix)

    outer_xsdata = openmc.XSdata('mat3', groups)
    outer_xsdata.order = 0
    outer_xsdata.set_total([outer_xs])
    outer_xsdata.set_absorption([0])
    outer_xsdata.set_scatter_matrix(scatter_matrix)

    one_g_XS_file = openmc.MGXSLibrary(groups)
    one_g_XS_file.add_xsdatas([mat1_xsdata, mat2_xsdata, outer_xsdata])
    one_g_XS_file.export_to_hdf5('xs.h5')


    # Create a dummy material with fictitious cross-section data
    mat1 = openmc.Material(name="Material1")
    mat1.add_macroscopic('mat1')

    mat2 = openmc.Material(name="Material2")
    mat2.add_macroscopic('mat2')

    mat_outer = openmc.Material(name="Material3")
    mat_outer.add_macroscopic('mat3')

    materials = openmc.Materials([mat1, mat2, mat_outer])
    materials.cross_sections = 'xs.h5'
    materials.export_to_xml()

    sphere_inner_left = openmc.Sphere(x0=-left_cell_offset, r=mat1_r, boundary_type='transmission')
    sphere_inner_right = openmc.Sphere(x0=right_cell_offset, r=mat2_r, boundary_type='transmission')
    sphere_outer = openmc.Sphere(r=outer_r, boundary_type='vacuum')

    cell_inner_left = openmc.Cell(name="Inner Sphere Left", fill=mat1, region=-sphere_inner_left)
    cell_inner_right = openmc.Cell(name="Inner Sphere Right", fill=mat2, region=-sphere_inner_right)
    cell_outer = openmc.Cell(name="Outer Sphere", fill=mat_outer, region=+sphere_inner_left & +sphere_inner_right & -sphere_outer)

    geometry = openmc.Geometry([cell_inner_left, cell_inner_right, cell_outer])
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

def test_optical_thickness_variations(lib_init, setup_vars):
    """Test optical thickness calculation for various scenarios."""
    mat1_xs = setup_vars["mat1_xs"]
    mat2_xs = setup_vars["mat2_xs"]
    outer_xs = setup_vars["outer_xs"]
    mat1_r = setup_vars["mat1_r"]
    mat2_r = setup_vars["mat2_r"]
    outer_r = setup_vars["outer_r"]
    left_cell_offset = setup_vars["left_cell_offset"]
    right_cell_offset = setup_vars["right_cell_offset"]

    # Test Case 1: Origin to left center (through mat1 and outer region)
    left_center_coord = -left_cell_offset
    expected_tau_left = mat1_xs * mat1_r + outer_xs * (abs(left_center_coord) - mat1_r)
    calculated_tau_left = openmc.lib.calculate_optical_thickness((0, 0, 0), (left_center_coord, 0, 0))
    assert np.isclose(calculated_tau_left, expected_tau_left), f"Failed for left center path"

    # Test Case 2: Origin to outer boundary in x-direction
    expected_tau_x = mat2_xs * (2 * mat2_r) + outer_xs * (outer_r - 2*mat2_r)
    calculated_tau_x = openmc.lib.calculate_optical_thickness((0, 0, 0), (outer_r, 0, 0))
    assert np.isclose(calculated_tau_x, expected_tau_x), f"Failed for outer boundary x path"

    # Test Case 3: Vertical path through outer region
    expected_tau_outer = outer_xs * (2 * outer_r)
    calculated_tau_outer = openmc.lib.calculate_optical_thickness((0, -outer_r, 0), (0, outer_r, 0))
    assert np.isclose(calculated_tau_outer, expected_tau_outer), f"Failed for outer region vertical path"

    # Test Case 4: Through both inner spheres and outer
    expected_tau_full = mat1_xs * (2 * mat1_r) + mat2_xs * (2 * mat2_r) + outer_xs * (2*outer_r - 2 * mat1_r - 2 * mat2_r)
    calculated_tau_full = openmc.lib.calculate_optical_thickness((-outer_r, 0, 0), (outer_r, 0, 0))
    assert np.isclose(calculated_tau_full, expected_tau_full), f"Failed for full traversal path"


def test_voxel_optical_thickness(lib_init):
    """Test voxel-based optical thickness calculation does not throw an error."""
    mesh = RegularMesh()
    mesh.dimension = (2, 2, 2)
    mesh.lower_left = (0, 0, 0)
    mesh.upper_right = (5, 5, 5)

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