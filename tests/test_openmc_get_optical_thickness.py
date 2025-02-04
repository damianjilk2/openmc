import openmc
from openmc import RegularMesh
import openmc.lib
import numpy as np

water = openmc.Material(name="h2o")
water.add_nuclide('H1', 2.0)
water.add_nuclide('O16', 1.0)
water.set_density('g/cm3', 1.0)

# Material 2: Uranium
material_u = openmc.Material(name="Uranium")
material_u.add_element('U', 1.0)
material_u.set_density('g/cm3', 19.1)

materials = openmc.Materials([water, material_u])
materials.export_to_xml()

# Geometry: define surfaces and cells
sphere_inner = openmc.Sphere(r=2, boundary_type='transmission')
sphere_outer = openmc.Sphere(r=10, boundary_type='vacuum')

# Inner cell (filled with water)
cell_inner = openmc.Cell(name="Inner Sphere", fill=water, region=-sphere_inner)

# Outer shell (filled with Uranium)
cell_outer = openmc.Cell(name="Outer Sphere", fill=material_u, region=+sphere_inner & -sphere_outer)

geometry = openmc.Geometry([cell_inner, cell_outer])
geometry.export_to_xml()

# Settings
settings = openmc.Settings()
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
            # TODO: what should happen when start_voxel = end_voxel?

            tau[start_voxel, end_voxel] = openmc.lib.get_optical_thickness(
                start_min, start_max, end_min, end_max, num_rays
            )

    openmc.lib.finalize()

    return tau

mesh = RegularMesh()
mesh.dimension = (2, 2, 2)  # Mesh resolution
mesh.lower_left = (0, 0, 0)
mesh.upper_right = (5.0, 5.0, 5.0)

num_rays = 10000
tau = calculate_optical_thickness_for_voxels(mesh, num_rays)

import csv
filename = "tests/tau.csv"
with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([''] + [f"Voxel {i}" for i in range(len(tau))])
        for i in range(len(tau)):
             writer.writerow([f"Voxel {i}"] + tau[i].tolist())
print(f"Matrix saved to {filename}")