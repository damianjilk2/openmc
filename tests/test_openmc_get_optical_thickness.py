import openmc
from openmc import RegularMesh
import openmc.lib
import numpy as np

# Material
material = openmc.Material(name="Hydrogen")
material.add_element('H', 1.0)
material.set_density('g/cm3', 0.071)
materials = openmc.Materials([material])
materials.export_to_xml()

# Geometry
sphere = openmc.Sphere(r=10, boundary_type='vacuum')
cell = openmc.Cell(fill=material, region=-sphere)
geometry = openmc.Geometry([cell])
geometry.export_to_xml()

# Settings
settings = openmc.Settings()
settings.particles = 1
settings.batches = 1
settings.export_to_xml()

def calculate_optical_thickness_for_voxels(mesh: RegularMesh, num_rays: int):
    openmc.lib.init()

    lower_left = np.array(mesh.lower_left)
    upper_right = np.array(mesh.upper_right)
    dimensions = np.array(mesh.dimension)

    # Calculate voxel size in each dimension
    voxel_size = (upper_right - lower_left) / dimensions

    # Preallocate optical thickness matrix
    num_voxels = np.prod(dimensions)
    print(num_voxels)
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
            # TODO: investigate start_voxel and end_voxel logic. Does it makes sense to only solve upper triangle?
            # TODO: also, what should happen when start_voxel = end_voxel?
            # TODO: should the tau matrix be symmetric?

            tau[start_voxel, end_voxel] = openmc.lib.get_optical_thickness(
                start_min, start_max, end_min, end_max, num_rays
            )

    openmc.lib.finalize()

    return tau

mesh = RegularMesh()
mesh.dimension = (1, 2, 2)  # mesh resolution
mesh.lower_left = (0.0, 0.0, 0.0)
mesh.upper_right = (5.0, 5.0, 5.0)

num_rays = 10
tau = calculate_optical_thickness_for_voxels(mesh, num_rays)
print(tau)