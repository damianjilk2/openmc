# test_slab.py
import numpy as np
import openmc
import pytest
from collision_probability import build_1d_collision_probability_matrix
from adjoint_flux import calculate_adjoint_flux, generate_weight_windows

def create_slab_model():
    """Create a simple 1D slab model."""
    fuel_mat = openmc.Material(name="Fuel")
    fuel_mat.set_density('g/cm3', 10.0)
    fuel_mat.add_nuclide('U235', 1.0)
    
    moderator_mat = openmc.Material(name="Moderator")
    moderator_mat.set_density('g/cm3', 1.0)
    moderator_mat.add_nuclide('H1', 2.0)
    moderator_mat.add_nuclide('O16', 1.0)
    
    materials = openmc.Materials([fuel_mat, moderator_mat])
    materials.export_to_xml()
    
    xplanes = [openmc.XPlane(x0=x) for x in np.linspace(0, 20, 6)]
    for plane in xplanes:
        plane.boundary_type = 'transmission'
    xplanes[0].boundary_type = 'vacuum'
    xplanes[-1].boundary_type = 'vacuum'
    
    regions = []
    cells = []
    
    # Fuel in regions 0 and 2, moderator in regions 1, 3, and 4
    materials_pattern = [fuel_mat, moderator_mat, fuel_mat, moderator_mat, moderator_mat]
    
    for i in range(5):
        region = -xplanes[i+1] & +xplanes[i]
        regions.append(region)
        cell = openmc.Cell(fill=materials_pattern[i], region=region)
        cells.append(cell)
    
    geometry = openmc.Geometry(cells)
    geometry.export_to_xml()
    
    settings = openmc.Settings()
    settings.particles = 1000
    settings.batches = 10
    settings.inactive = 5
    settings.export_to_xml()
    
    # Define regions for collision probability calculation
    # List of (sigma, width) tuples
    cp_regions = [
        (1.0, 4.0),    # Region 0: Fuel
        (0.5, 4.0),    # Region 1: Moderator
        (1.0, 4.0),    # Region 2: Fuel
        (0.5, 4.0),    # Region 3: Moderator
        (0.5, 4.0)     # Region 4: Moderator
    ]
    
    return cp_regions

def test_slab_collision_probability():
    """Test collision probability calculation for 1D slab."""
    # Create slab model
    cp_regions = create_slab_model()
    
    # Calculate collision probability matrix
    P = build_1d_collision_probability_matrix(cp_regions)
    
    # Check if P is a valid probability matrix
    assert np.all(P >= 0) and np.all(P <= 1), "Invalid probability values in matrix"
    
    # Check if row sums are close to 1 (conservation principle)
    row_sums = np.sum(P, axis=1)
    assert np.allclose(row_sums, 1.0, rtol=0.2), "Conservation principle violated"
    
    # Calculate scattering ratios (simplified for testing)
    C = np.array([0.8, 0.9, 0.8, 0.9, 0.9])
    
    # Define adjoint source (simplified response function)
    adjoint_source = np.zeros(5)
    adjoint_source[-1] = 1.0  # Source at the last region
    
    # Calculate adjoint flux
    phi_adjoint = calculate_adjoint_flux(P, C, adjoint_source)
    
    # Check if adjoint flux is non-negative and decreasing from source
    assert np.all(phi_adjoint >= 0), "Negative adjoint flux values"
    assert np.all(np.diff(phi_adjoint[::-1]) <= 0), "Adjoint flux should decrease away from source"
    
    # Generate weight windows
    w_lower, w_upper = generate_weight_windows(phi_adjoint)
    
    # Check if weight windows are valid
    assert np.all(w_lower > 0), "Lower weight bound must be positive"
    assert np.all(w_upper > w_lower), "Upper weight bound must exceed lower bound"
    
    print("Collision Probability Matrix:")
    print(P)
    print("\nAdjoint Flux:")
    print(phi_adjoint)
    print("\nWeight Windows:")
    print("Lower bounds:", w_lower)
    print("Upper bounds:", w_upper)
    
    return P, phi_adjoint, w_lower, w_upper