# analysis.py

def calculate_figure_of_merit(variance: float, computation_time: float) -> float:
    """
    Calculate Figure of Merit (FOM) for comparing methods.
    
    FOM = 1 / (variance * computation_time)
    
    Parameters
    ----------
    variance : float
        Variance of the estimator
    computation_time : float
        Computation time in seconds
        
    Returns
    -------
    float
        Figure of Merit value
    """
    return 1.0 / (variance * computation_time)