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
    S = np.diag(source)

    if method == 'direct':
        # direct matrix inversion phi = V^-1 (sigma_t - P sigma_s)^-1 P V S
        matrix = sigma_t - P @ sigma_s

        phi = np.linalg.inv(V) @ np.linalg.inv(matrix) @ P @ V @ S

        # cond_forward = np.linalg.cond(matrix)
        # cond_P = np.linalg.cond(P)
        # print(f"Condition number of forward matrix: {cond_forward}")
        # print(f"Condition number of P: {cond_P}")
        # print(f'||P||2 (norm of P): {np.linalg.norm(P, 2)}')
        # print(f'||P - P^T||2 (asymmetry of P): {np.linalg.norm(P - P.T, 2)}')
        
    elif method == 'iterative':
        # NOTE: iterative method has not been updated yet.

        # iterative series solution phi = P Q + (P sigma V P Q) + (P sigma V P sigma V P Q) + ...
        # phi = sum (I + P sigma V)^n P Q
        phi = P @ Q
        print(f"\n=== Iterative Forward Flux Solver ===")
        print(f"Initial phi = P @ Q:")
        print(phi)

        norm_initial = np.linalg.norm(phi)
        print(f"Initial norm: {norm_initial:.6e}")

        # alpha = 1.0 / (np.linalg.norm(P @ sigma_V, 2) + 1e-6)

        for i in range(max_iter):
            # phi_{n+1} = P Q + P sigma V phi_n
            phi_new = P @ Q + P @ sigma_V @ phi
            # phi_new = (1 - alpha) * phi + alpha * (P @ Q + P @ sigma_V @ phi)

            # check convergence
            if np.linalg.norm(phi_new - phi) / np.linalg.norm(phi_new) < tol:
                return phi_new
            
            phi = phi_new

        print(f"Failed to converge in {max_iter} iterations")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return phi

def calculate_adjoint_flux(P: np.ndarray, sigma: np.ndarray, volumes: np.ndarray, Q_adj: np.ndarray, method: str = 'iterative',
                       max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """
    Solve adjoint flux equation.
    
    Parameters
    ----------
    P : np.ndarray
        Collision probability matrix [n x n]
    sigma : np.ndarray
        Cross sections vector [n]
    volumes : np.ndarray
        Volumes of each region [n]
    Q_adj : np.ndarray
        Adjoint source vector [n]
    method : str
        'iterative' (default) or 'direct'
        
    Returns
    -------
    phi_adj : np.ndarray
        Computed adjoint flux vector [n]
    """
    sigma_V = np.diag(sigma * volumes)

    if method == 'direct':
        # direct solution: phi_adj = inv(I - P^T sigma V) P^T Q_adj
        I = np.eye(len(Q_adj))
        matrix = I - P.T @ sigma_V
        phi_adj = np.linalg.inv(matrix) @ P.T @ Q_adj
        
    elif method == 'iterative':
        phi_adj = P.T @ Q_adj
        for _ in range(max_iter):
            # phi*_{n+1} = P^T Q_adj + P^T sigma V phi*_n
            phi_new = P.T @ Q_adj + P.T @ sigma_V @ phi_adj

            # check convergence
            if np.linalg.norm(phi_new - phi_adj) / np.linalg.norm(phi_new) < tol:
                return phi_new
            
            phi_adj = phi_new
        
        print(f"Failed to converge in {max_iter} iterations")
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