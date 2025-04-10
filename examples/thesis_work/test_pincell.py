# test_pincell.py
import numpy as np
import openmc
import pytest
from collision_probability import calculate_simplified_3d_collision_probabilities, get_voxel_properties
from adjoint_flux import calculate_adjoint_flux, generate_weight_windows, get_scattering_ratios

def create_pincell_model():
    """Create a simple pin cell model."""
    fuel = openmc.Material(name='Fuel')
    fuel.set_density('g/cm3', 10.0)
    fuel.add_nuclide('U235', 1.0)
    
    clad = openmc.Material(name='Cladding')
    clad.set_density('g/cm3', 6.0)
    clad.add_nuclide('Zr90', 1.0)
    
    water = openmc.Material(name='Water')
    water.set_density('g/cm3', 1.0)
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    
    materials = openmc.Materials([fuel, clad, water])
    materials.export_to_xml()
    
    fuel_radius = 0.4
    clad_inner_radius = 0.41
    clad_outer_radius = 0.47
    pitch = 1.26
    
    fuel_surf = openmc.ZCylinder(r=fuel_radius)
    clad_inner_surf = openmc.ZCylinder(r=clad_inner_radius)
    clad_outer_surf = openmc.ZCylinder(r=clad_outer_radius)
    
    min_z = openmc.ZPlane(z0=-1.0, boundary_type='reflective')
    max_z = openmc.ZPlane(z0=1.0, boundary_type='reflective')
    min_x = openmc.XPlane(x0=-pitch/2, boundary_type='reflective')
    max_x = openmc.XPlane(x0=pitch/2, boundary_type='reflective')
    min_y = openmc.YPlane(y0=-pitch/2, boundary_type='reflective')
    max_y = openmc.YPlane(y0=pitch/2, boundary_type='reflective')
    
    fuel_cell = openmc.Cell(name='fuel')
    fuel_cell.fill = fuel
    fuel_cell.region = -fuel_surf & +min_z & -max_z
    
    gap_cell = openmc.Cell(name='gap')
    gap_cell.region = +fuel_surf & -clad_inner_surf & +min_z & -max_z
    
    clad_cell = openmc.Cell(name='clad')
    clad_cell.fill = clad
    clad_cell.region = +clad_inner_surf & -clad_outer_surf & +min_z & -max_z
    
    water_cell = openmc.Cell(name='water')
    water_cell.fill = water
    water_cell.region = +clad_outer_surf & +min_x & -max_x & +min_y & -max_y & +min_z & -max_z
    
    geometry = openmc.Geometry([fuel_cell, gap_cell, clad_cell, water_cell])
    geometry.export_to_xml()
    
    settings = openmc.Settings()
    settings.particles = 1000
    settings.batches = 10
    settings.inactive = 5
    
    bounds = [-pitch/2, pitch/2, -pitch/2, pitch/2, -1.0, 1.0]
    uniform_dist = openmc.stats.Box(bounds[:2], bounds[2:4], bounds[4:])
    settings.source = openmc.Source(space=uniform_dist)
    settings.export_to_xml()
    
    mesh = openmc.RegularMesh()
    mesh.dimension = [10, 10, 2]
    mesh.lower_left = [-pitch/2, -pitch/2, -1.0]
    mesh.upper_right = [pitch/2, pitch/2, 1.0]
    
    return mesh

def test_pincell_collision_probability():
    """Test collision probability calculation for pin cell."""
    mesh = create_pincell_model()
    
    openmc.lib.init()
    
    try:
        P = calculate_simplified_3d_collision_probabilities(mesh, num_rays=50)
        
        # check if P is a valid probability matrix
        assert np.all(P >= 0) and np.all(P <= 1), "Invalid probability values in matrix"
        
        C = get_scattering_ratios(mesh)
        
        # define adjoint source (using last few voxels as detector)
        n_voxels = np.prod(mesh.dimension)
        adjoint_source = np.zeros(n_voxels)
        adjoint_source[-10:] = 1.0  # Last 10 voxels as detector region
        
        phi_adjoint = calculate_adjoint_flux(P, C, adjoint_source)
        
        # check if adjoint flux is non-negative
        assert np.all(phi_adjoint >= 0), "Negative adjoint flux values"
        
        w_lower, w_upper = generate_weight_windows(phi_adjoint)
        
        # check if weight windows are valid
        assert np.all(w_lower > 0), "Lower weight bound must be positive"
        assert np.all(w_upper > w_lower), "Upper weight bound must exceed lower bound"
        
        print(f"Pin cell model with {n_voxels} voxels")
        print(f"Maximum adjoint flux: {np.max(phi_adjoint)}")
        print(f"Minimum adjoint flux: {np.min(phi_adjoint)}")
        print(f"Weight window ranges: [{np.min(w_lower)}, {np.max(w_upper)}]")
        
        return P, phi_adjoint, w_lower, w_upper
    
    finally:
        openmc.lib.finalize()