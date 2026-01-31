import numpy as np
from scipy.spatial import ConvexHull

def unwrap_periodic(pos, box_size):
    """
    Unwraps particle positions relative to their center of mass 
    to handle periodic boundary conditions for shape analysis.
    
    Args:
        pos: (N, 3) or (N, 2) array of positions
        box_size: float, size of the simulation box
        
    Returns:
        pos_unwrapped: positions adjusted to be continuous
    """
    pos = np.asarray(pos, dtype=float)
    com = pos.mean(axis=0)
    rel = pos - com
    rel -= box_size * np.round(rel / box_size)
    return com + rel


def pca_axes(pos, box_size=None):
    """
    Computes principal axes lengths of the flock using PCA.
    
    Args:
        pos: (N, 3) array of positions
        box_size: optional float. If provided, positions are unwrapped first.
        
    Returns:
        (L1, L2, L3): Principal lengths sorted descending (L1 >= L2 >= L3)
    """
    if box_size is not None:
        pos = unwrap_periodic(pos, box_size)
    
    # Center the positions
    X = pos - pos.mean(axis=0)
    
    # Covariance matrix
    # divide by N or N-1? Using max(shape[0], 1) like in original code
    C = (X.T @ X) / max(X.shape[0], 1)
    
    # Eigendecomposition
    evals, _ = np.linalg.eigh(C)
    evals = np.sort(evals)[::-1] # Descending
    evals = np.maximum(evals, 0.0)
    
    # Principal lengths (sqrt of eigenvalues)
    L = np.sqrt(evals)
    
    # Handle cases with fewer dimensions (e.g. 2D) gracefully if needed, 
    # but here assuming 3D return. Safe to return as many as we have + 0s?
    # Original code hardcoded 3 dimensions.
    if len(L) >= 3:
        return float(L[0]), float(L[1]), float(L[2])
    else:
        # Fallback for 2D or edge cases
        ret = [0.0, 0.0, 0.0]
        for i in range(min(len(L), 3)):
            ret[i] = float(L[i])
        return tuple(ret)


def flock_volume_from_pca(pos, box_size=None):
    """
    Approximates flock volume using PCA-derived ellipsoid.
    
    Args:
        pos: (N, 3) array of positions
        box_size: optional float for unwrapping
    
    Returns:
        V: float, Volume = (4/3) * pi * L1 * L2 * L3
        (L1, L2, L3): The principal axes lengths
    """
    L1, L2, L3 = pca_axes(pos, box_size=box_size)
    V = (4.0 / 3.0) * np.pi * L1 * L2 * L3
    return float(V), (L1, L2, L3)


def hull_volume(pos, box_size=None):
    """
    Computes volume of the Convex Hull of the positions.
    Returns 0.0 if fewer than 4 points or if calculation fails.
    
    Args:
        pos: (N, 3) array
        box_size: optional float. If provided, positions are unwrapped.
                  (Usually hull needs continuous points too)
    """
    pos = np.asarray(pos, dtype=float)
    if box_size is not None:
        pos = unwrap_periodic(pos, box_size)
        
    if pos.shape[0] < 4:
        return 0.0
    try:
        hull = ConvexHull(pos)
        return float(hull.volume)
    except Exception:
        return 0.0
