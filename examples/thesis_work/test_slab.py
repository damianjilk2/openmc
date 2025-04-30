# test_slab.py
import numpy as np
import openmc
import openmc.lib
import openmc.mgxs as mgxs
import pytest
from collision_probability import calculate_3d_collision_probability_matrix
from flux import calculate_forward_flux, calculate_adjoint_flux, generate_weight_windows

SLAB_GEOMETRY_CONFIG = {
    'materials': [
        {'name': 'fuel', 'sigma_total': 0.1, 'width': 10.0},
        {'name': 'moderator', 'sigma_total': 0.05, 'width': 10.0},
    ],
    'mesh_dimension': [2, 1, 1],
    'boundary_types': ['vacuum', 'transmission', 'vacuum']
}

def create_slab_model():
    """Create a simple 1D slab model using macroscopic XS data."""
    groups = mgxs.EnergyGroups([1e-5, 1e6])
    scatter_matrix = np.array([[[0]]])

    # Create macroscopic cross sections for each material
    mgxs_lib = openmc.MGXSLibrary(groups)
    materials_list = []
    
    for mat_config in SLAB_GEOMETRY_CONFIG['materials']:
        xs = openmc.XSdata(mat_config['name'], groups)
        xs.order = 0
        xs.set_total([mat_config['sigma_total']])
        xs.set_absorption([0.0])
        xs.set_scatter_matrix(scatter_matrix)
        mgxs_lib.add_xsdata(xs)
        
        material = openmc.Material(name=mat_config['name'])
        material.add_macroscopic(mat_config['name'])
        materials_list.append(material)

    mgxs_lib.export_to_hdf5('xs.h5')

    materials = openmc.Materials(materials_list)
    materials.cross_sections = 'xs.h5'
    materials.export_to_xml()

    # Create geometry planes with configured boundaries
    total_width = sum(mat['width'] for mat in SLAB_GEOMETRY_CONFIG['materials'])
    boundary_types = SLAB_GEOMETRY_CONFIG['boundary_types']
    
    planes = []
    current_x = 0.0
    planes.append(openmc.XPlane(x0=current_x, boundary_type=boundary_types[0]))
    
    for mat in SLAB_GEOMETRY_CONFIG['materials'][:-1]:
        current_x += mat['width']
        planes.append(openmc.XPlane(x0=current_x, boundary_type=boundary_types[1]))
    
    current_x += SLAB_GEOMETRY_CONFIG['materials'][-1]['width']
    planes.append(openmc.XPlane(x0=current_x, boundary_type=boundary_types[2]))

    # Create cells for each material region
    cells = []
    for i, mat_config in enumerate(SLAB_GEOMETRY_CONFIG['materials']):
        region = +planes[i] & -planes[i+1]
        cell = openmc.Cell(
            fill=materials_list[i],
            region=region,
            name=f"{mat_config['name']}_cell"
        )
        cells.append(cell)

    geometry = openmc.Geometry(cells)
    geometry.export_to_xml()

    settings = openmc.Settings()
    settings.energy_mode = "multi-group"
    settings.particles = 1
    settings.batches = 1
    settings.export_to_xml()

    mesh = openmc.RegularMesh()
    mesh.dimension = SLAB_GEOMETRY_CONFIG['mesh_dimension']
    mesh.lower_left = [0.0, 0.0, 0.0]
    total_width = sum(mat['width'] for mat in SLAB_GEOMETRY_CONFIG['materials'])
    mesh.upper_right = [total_width, 1.0, 1.0]

    return mesh

def test_slab_collision_probability():
    """Test collision probability calculation for 1D slab."""
    mesh = create_slab_model()
    openmc.lib.init()
    
    # extract cross sections and widths from configuration
    cp_regions = [
        (mat['sigma_total'], mat['width']) 
        for mat in SLAB_GEOMETRY_CONFIG['materials']
    ]
    sigmas = np.array([region[0] for region in cp_regions])
    widths = np.array([region[1] for region in cp_regions])
    volumes = widths  # 1D volumes are widths

    forward_source = np.array([1.0, 0.0])
    adjoint_source = np.array([0.0, 1.0])
    
    P = calculate_3d_collision_probability_matrix(mesh, volumes, sigmas, num_rays=200)

    assert np.all(P >= 0) and np.all(P <= 1), "Invalid probability values in matrix"
    
    # conservation principle
    row_sums = np.sum(P, axis=1)
    assert np.allclose(row_sums, 1.0, rtol=0.3), "Conservation principle violated"
    
    forward_flux = calculate_forward_flux(P, sigmas, volumes, forward_source, method='direct')
    print("\nForward Flux:")
    print(forward_flux)
    
    adjoint_flux = calculate_adjoint_flux(P, sigmas, volumes, adjoint_source, method='iterative', max_iter = 100000)
    print("\nAdjoint Flux:")
    print(adjoint_flux)
    
    assert np.all(forward_flux >= 0), "Negative forward flux values"
    assert np.all(adjoint_flux >= 0), "Negative adjoint flux values"
    
    w_lower, w_upper = generate_weight_windows(adjoint_flux, alpha=1.0, C_w=5.0)
    
    # Check if weight windows are valid
    assert np.all(w_lower > 0), "Lower weight bound must be positive"
    assert np.all(w_upper > w_lower), "Upper weight bound must exceed lower bound"
    
    print("\nCollision Probability Matrix:")
    print(P)
    print("\nWeight Windows:")
    print("Lower bounds:", w_lower)
    print("Upper bounds:", w_upper)

    openmc.lib.finalize()
    
    return P, forward_flux, adjoint_flux, w_lower, w_upper

if __name__ == "__main__":
    slab_results = test_slab_collision_probability()