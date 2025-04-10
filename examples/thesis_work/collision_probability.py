# collision_probability.py
import numpy as np
import openmc
import openmc.lib
from typing import Tuple, List, Optional
from scipy import special
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

def calculate_exponential_integral(n: int, x: float) -> float:
    """
    Calculate the exponential integral E_n(x).
    
    Parameters
    ----------
    n : int
        Order of the exponential integral
    x : float
        Argument value
        
    Returns
    -------
    float
        Value of the exponential integral E_n(x)
    """
    return special.expn(n, x)

def calculate_1d_self_collision_probability(sigma: float, delta: float) -> float:
    """
    Calculate self-collision probability for a 1D region.
    
    Parameters
    ----------
    sigma : float
        Total cross section
    delta : float
        Region width
        
    Returns
    -------
    float
        Self-collision probability P_ii
    """
    tau = sigma * delta
    return 1.0 - (1.0 / (2.0 * tau)) * (1.0 - 2.0 * calculate_exponential_integral(3, tau))

def calculate_1d_collision_probability(sigma_i: float, sigma_j: float, 
                                     delta_i: float, delta_j: float, 
                                     tau_ij: float) -> float:
    """
    Calculate collision probability between different regions in 1D.
    
    Parameters
    ----------
    sigma_i : float
        Total cross section in region i
    sigma_j : float
        Total cross section in region j
    delta_i : float
        Width of region i
    delta_j : float
        Width of region j
    tau_ij : float
        Optical path length between regions
        
    Returns
    -------
    float
        Collision probability P_ij
    """
    term1 = calculate_exponential_integral(3, tau_ij)
    term2 = calculate_exponential_integral(3, tau_ij + sigma_i * delta_i)
    term3 = calculate_exponential_integral(3, tau_ij + sigma_j * delta_j)
    term4 = calculate_exponential_integral(3, tau_ij + sigma_i * delta_i + sigma_j * delta_j)
    
    return (1.0 / (2.0 * sigma_i * delta_i)) * (term1 - term2 - term3 + term4)

def build_1d_collision_probability_matrix(regions: List[Tuple[float, float]]) -> np.ndarray:
    """
    Build the collision probability matrix for 1D geometry.
    
    Parameters
    ----------
    regions : List[Tuple[float, float]]
        List of (sigma, width) tuples for each region
        
    Returns
    -------
    np.ndarray
        Collision probability matrix P
    """
    n_regions = len(regions)
    P = np.zeros((n_regions, n_regions))
    
    # self-collision probabilities
    for i in range(n_regions):
        sigma_i, delta_i = regions[i]
        P[i, i] = calculate_1d_self_collision_probability(sigma_i, delta_i)
    
    # collision probabilities between different regions
    for i in range(n_regions):
        sigma_i, delta_i = regions[i]
        
        # position of region boundaries
        x = np.zeros(n_regions + 1)
        x[0] = 0.0
        for k in range(n_regions):
            x[k+1] = x[k] + regions[k][1]
        
        for j in range(n_regions):
            if i != j:
                sigma_j, delta_j = regions[j]
                
                # optical path between regions
                if j > i:
                    tau_ij = sum(regions[k][0] * regions[k][1] for k in range(i+1, j))
                else:  # j < i
                    tau_ij = sum(regions[k][0] * regions[k][1] for k in range(j+1, i))
                
                P[i, j] = calculate_1d_collision_probability(sigma_i, sigma_j, delta_i, delta_j, tau_ij)
    
    return P

def calculate_simplified_3d_collision_probabilities(mesh, num_rays: int = 100) -> np.ndarray:
    """
    Calculate collision probabilities for a 3D mesh using ray tracing.
    
    Parameters
    ----------
    mesh : openmc.RegularMesh
        Mesh dividing the geometry into voxels
    num_rays : int, optional
        Number of rays to use for Monte Carlo estimation
        
    Returns
    -------
    np.ndarray
        Collision probability matrix
    """
    tau = calculate_optical_thickness_for_voxels(mesh, num_rays)
    
    volumes, cross_sections = get_voxel_properties(mesh)
    
    n_voxels = np.prod(mesh.dimension)
    
    # Initialize collision probability matrix
    P = np.zeros((n_voxels, n_voxels))

    # calculate voxel centers once
    centers = np.array([get_voxel_center(mesh, i) for i in range(n_voxels)])
    
    # collision probabilities built from 1d
    for i in range(n_voxels):
        for j in range(n_voxels):
            if i == j:
                # self-collision (approximation based on mean chord length)
                # Mean chord length for a convex body: L = 4V/S (V=volume, S=surface area)
                # For a cube with side length a: L = 4a^3/(6a^3) = 2a/3
                voxel_size = np.cbrt(volumes[i])  # cube root of volume
                mean_chord = 2 * voxel_size / 3
                
                # Escape probability: exp(-sigma*L)
                escape_prob = np.exp(-cross_sections[i] * mean_chord)
                
                # Self-collision probability is (1 - escape probability)
                P[i, i] = 1.0 - escape_prob
            else:
                # collision between different voxels
                dist = np.linalg.norm(centers[i] - centers[j])
                
                solid_angle_factor = volumes[j] / (4 * np.pi * dist**2)
                
                P[i, j] = cross_sections[i] * solid_angle_factor * np.exp(-tau[i, j])
    
    # normalize to satisfy the conservation principle
    for j in range(n_voxels):
        P[:, j] /= np.sum(P[:, j])
    
    #TODO: from this very simplified solution, investigate the need for more rigor (2d integral presented in L+M with ray tracing)
    return P

def calculate_optical_thickness_for_voxels(mesh, num_rays: int) -> np.ndarray:
    """
    Calculate optical thickness between voxels in a mesh.
    
    Parameters
    ----------
    mesh : openmc.RegularMesh
        Mesh dividing the geometry into voxels
    num_rays : int
        Number of rays to use for Monte Carlo estimation
        
    Returns
    -------
    np.ndarray
        Optical thickness matrix
    """
    lower_left = np.array(mesh.lower_left)
    upper_right = np.array(mesh.upper_right)
    dimensions = np.array(mesh.dimension)
    
    voxel_size = (upper_right - lower_left) / dimensions
    
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

def get_voxel_bounds(mesh, voxel_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the minimum and maximum bounds of a voxel.
    
    Parameters
    ----------
    mesh : openmc.RegularMesh
        Mesh containing the voxels
    voxel_idx : int
        Index of the voxel
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (min_bound, max_bound) of the voxel
    """
    dimensions = np.array(mesh.dimension)
    lower_left = np.array(mesh.lower_left)
    upper_right = np.array(mesh.upper_right)
    
    voxel_size = (upper_right - lower_left) / dimensions
    
    # linear index to 3D indices
    i = voxel_idx % dimensions[0]
    j = (voxel_idx // dimensions[0]) % dimensions[1]
    k = voxel_idx // (dimensions[0] * dimensions[1])
    
    min_bound = lower_left + np.array([i, j, k]) * voxel_size
    max_bound = min_bound + voxel_size
    
    return min_bound, max_bound

def get_voxel_properties(mesh):
    """
    Get volumes and cross sections for each voxel in a mesh.
    
    Parameters
    ----------
    mesh : openmc.RegularMesh
        Mesh dividing the geometry into voxels
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Volumes and cross sections for each voxel
    """
    dimensions = mesh.dimension
    lower_left = mesh.lower_left
    upper_right = mesh.upper_right
    
    voxel_size = [(upper_right[i] - lower_left[i]) / dimensions[i] for i in range(3)]
    voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]
    
    n_voxels = np.prod(dimensions)
    volumes = np.ones(n_voxels) * voxel_volume
    cross_sections = np.zeros(n_voxels)
    
    for i in range(n_voxels):
        center = get_voxel_center(mesh, i)
        _, cell = openmc.lib.find_cell(center)
        if cell is not None:
            material = cell.fill
            if material is not None:
                cross_sections[i] = material.get_total_xs()
    
    return volumes, cross_sections

def get_voxel_center(mesh, voxel_index):
    """
    Get the center coordinates of a voxel.
    
    Parameters
    ----------
    mesh : openmc.RegularMesh
        Mesh dividing the geometry into voxels
    voxel_index : int
        Index of the voxel
        
    Returns
    -------
    np.ndarray
        Center coordinates of the voxel
    """
    dimensions = np.array(mesh.dimension)
    lower_left = np.array(mesh.lower_left)
    upper_right = np.array(mesh.upper_right)
    
    voxel_size = (upper_right - lower_left) / dimensions
    
    i = voxel_index % dimensions[0]
    j = (voxel_index // dimensions[0]) % dimensions[1]
    k = voxel_index // (dimensions[0] * dimensions[1])
    
    center = lower_left + (np.array([i, j, k]) + 0.5) * voxel_size
    
    return center

def get_voxel_surface_area(mesh, voxel_index):
    """
    Get the surface area of a voxel.
    
    Parameters
    ----------
    mesh : openmc.RegularMesh
        Mesh dividing the geometry into voxels
    voxel_index : int
        Index of the voxel
        
    Returns
    -------
    float
        Surface area of the voxel
    """
    dimensions = np.array(mesh.dimension)
    lower_left = np.array(mesh.lower_left)
    upper_right = np.array(mesh.upper_right)
    
    voxel_size = (upper_right - lower_left) / dimensions
    
    # For a rectangular voxel: 2*(dx*dy + dx*dz + dy*dz)
    return 2 * (voxel_size[0]*voxel_size[1] + 
               voxel_size[0]*voxel_size[2] + 
               voxel_size[1]*voxel_size[2])

def get_voxel_volume(mesh, voxel_index):
    """
    Get the volume of a voxel.
    
    Parameters
    ----------
    mesh : openmc.RegularMesh
        Mesh dividing the geometry into voxels
    voxel_index : int
        Index of the voxel
        
    Returns
    -------
    float
        Volume of the voxel
    """
    dimensions = np.array(mesh.dimension)
    lower_left = np.array(mesh.lower_left)
    upper_right = np.array(mesh.upper_right)
    
    voxel_size = (upper_right - lower_left) / dimensions
    
    return voxel_size[0] * voxel_size[1] * voxel_size[2]
