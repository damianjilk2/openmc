# adjoint_flux.py
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import openmc
from collision_probability import get_voxel_center

def calculate_adjoint_flux(P, C, adjoint_source):
    """
    Calculate the adjoint flux using collision probability method.
    
    Parameters
    ----------
    P : np.ndarray
        Collision probability matrix
    C : np.ndarray
        Diagonal matrix of scattering to total cross section ratios
    adjoint_source : np.ndarray
        Adjoint source term
        
    Returns
    -------
    np.ndarray
        Adjoint flux
    """
    # Convert to sparse matrices for efficiency
    P_sparse = sparse.csr_matrix(P)
    C_sparse = sparse.diags(C)
    
    # Calculate (I - CP^T)
    I = sparse.eye(P.shape[0])
    A = I - C_sparse @ P_sparse.T
    
    # Solve
    phi_adjoint = spla.spsolve(A, adjoint_source)
    
    return phi_adjoint

def generate_weight_windows(phi_adjoint, alpha=1.0, C_w=5.0):
    """
    Generate weight windows from adjoint flux.
    
    Parameters
    ----------
    phi_adjoint : np.ndarray
        Adjoint flux
    alpha : float, optional
        Normalization constant
    C_w : float, optional
        Weight window width factor
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Lower and upper weight window bounds
    """
    # lower weight window bounds
    w_lower = alpha / phi_adjoint
    
    # upper weight window bounds
    w_upper = C_w * w_lower
    
    return w_lower, w_upper

def get_scattering_ratios(mesh):
    """
    Get the scattering to total cross section ratios for each voxel.
    
    Parameters
    ----------
    mesh : openmc.RegularMesh
        Mesh dividing the geometry into voxels
        
    Returns
    -------
    np.ndarray
        Scattering to total cross section ratios
    """
    dimensions = mesh.dimension
    n_voxels = np.prod(dimensions)
    C = np.zeros(n_voxels)
    
    for i in range(n_voxels):
        center = get_voxel_center(mesh, i)
        _, cell = openmc.lib.find_cell(center)
        if cell is not None:
            material = cell.fill
            if material is not None:
                total_xs = material.get_total_xs()
                scatter_xs = material.get_scatter_xs()
                if total_xs > 0:
                    C[i] = scatter_xs / total_xs
    
    return C