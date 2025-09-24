import numpy as np

def accel(mu: float, point_r: np.ndarray, mass_r: np.ndarray = np.array([0.0, 0.0, 0.0])) -> np.ndarray:
    if np.array_equal(point_r, mass_r):
        raise ValueError("Singularity encountered")

    # Compute the relative vector
    rel_r = point_r - mass_r

    # Compute the magnitude
    mag = np.linalg.norm(rel_r)
    
    return -mu * rel_r / mag ** 3