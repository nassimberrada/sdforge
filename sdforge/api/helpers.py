import numpy as np

def coincident(point_a, point_b):
    """Checks if two points are coincident within a small tolerance."""
    return np.allclose(point_a, point_b, atol=1e-6)

def midpoint(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """Calculates the midpoint between two points."""
    return (np.array(point_a) + np.array(point_b)) / 2.0

def tangent_offset(circle_radius: float, line_direction: np.ndarray) -> np.ndarray:
    """Calculates the offset to make a line tangent to a circle."""
    perp_vec = np.array([-line_direction[1], line_direction[0], 0.0])
    return perp_vec * circle_radius