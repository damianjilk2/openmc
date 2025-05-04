# flux.py
import numpy as np

def calculate_forward_flux(P: np.ndarray, sigma_total: np.ndarray, sigma_scatter: np.ndarray, volumes: np.ndarray, source: np.ndarray, method: str = 'iterative',
                       max_iter: int = 1000, tol: float = 1e-3) -> np.ndarray:
    """
    Solve forward flux equation using either iterative or direct method. 
    
    Parameters
    ----------
    P : np.ndarray
        Collision probability matrix [n x n]
    sigma_total : np.ndarray
        Total cross sections vector [n]
    sigma_scatter : np.ndarray
        Scattering cross sections vector [n]
    volumes : np.ndarray
        Volumes of each region [n]
    source : np.ndarray 
        Source vector [n]
    method : str
        'iterative' (default) or 'direct'
        
    Returns
    -------
    phi : np.ndarray
        Computed flux vector [n]
    """
    V = np.diag(volumes)
    sigma_t = np.diag(sigma_total)
    sigma_s = np.diag(sigma_scatter)
    S = source

    if method == 'direct':
        # direct matrix inversion phi = V^-1 (sigma_t - P sigma_s)^-1 P V S
        matrix = sigma_t - P @ sigma_s
        phi = np.linalg.inv(V) @ np.linalg.inv(matrix) @ P @ V @ S
        
    elif method == 'iterative':
        # NOTE: iterative method has not been updated yet.
        pass
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return phi

def calculate_adjoint_flux(P: np.ndarray, sigma_total: np.ndarray, sigma_scatter: np.ndarray, volumes: np.ndarray, source_adj: np.ndarray, method: str = 'iterative',
                       max_iter: int = 1000, tol: float = 1e-3) -> np.ndarray:
    """
    Solve adjoint flux equation using either iterative or direct method. 
    
    Parameters
    ----------
    P : np.ndarray
        Collision probability matrix [n x n]
    sigma_total : np.ndarray
        Total cross sections vector [n]
    sigma_scatter : np.ndarray
        Scattering cross sections vector [n]
    volumes : np.ndarray
        Volumes of each region [n]
    source_adj : np.ndarray
        Adjoint source vector [n]
    method : str
        'iterative' (default) or 'direct'
        
    Returns
    -------
    phi_adj : np.ndarray
        Computed adjoint flux vector [n]
    """
    V = np.diag(volumes)
    sigma_t = np.diag(sigma_total)
    sigma_s = np.diag(sigma_scatter)
    S_adj = source_adj

    if method == 'direct':
        # direct matrix inversion phi_adj = V^-1 (sigma_t - P^T sigma_s)^-1 P^T V S_adj
        matrix = sigma_t - P.T @ sigma_s
        phi_adj = np.linalg.inv(V) @ np.linalg.inv(matrix) @ P.T @ V @ S_adj
        
    elif method == 'iterative':
        pass
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return phi_adj

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
    w_lower = alpha / phi_adjoint
    w_upper = C_w * w_lower
    return w_lower, w_upper