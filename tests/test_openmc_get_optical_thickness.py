import openmc
from openmc import RegularMesh
import openmc.lib
import numpy as np
import csv
import openmc.mgxs as mgxs
# import pytest

def define_fictitious_xs():
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

def define_mat_and_geom():
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

def define_settings():
    settings = openmc.Settings()
    settings.energy_mode = "multi-group"
    settings.particles = 1
    settings.batches = 1
    settings.export_to_xml()

def calculate_optical_thickness_for_voxels(mesh: RegularMesh, num_rays: int):
    openmc.lib.init()
    openmc.lib.simulation_init()

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
            print(f"startmin: {start_min}, startmax: {start_max}, endmin: {end_min}, endmax: {end_max}")
            # TODO: what should happen when start_voxel = end_voxel?

            tau[start_voxel, end_voxel] = openmc.lib.get_mean_optical_thickness_between_voxels(
                start_min, start_max, end_min, end_max, num_rays
            )
    return tau

if __name__ == "__main__":
    define_fictitious_xs()
    define_mat_and_geom()
    define_settings()

    mesh = RegularMesh()
    mesh.dimension = (2, 2, 2)  # Mesh resolution
    mesh.lower_left = (0, 0, 0)
    mesh.upper_right = (2, 2, 2)

    num_rays = 10000
    tau = calculate_optical_thickness_for_voxels(mesh, num_rays)

    filename = "tests/tau.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([''] + [f"Voxel {i}" for i in range(len(tau))])
        for i in range(len(tau)):
            writer.writerow([f"Voxel {i}"] + tau[i].tolist())
    print(f"Matrix saved to {filename}")

    print(openmc.lib.calculate_optical_thickness((0,0,0), (10,0,0)))

    openmc.lib.finalize()