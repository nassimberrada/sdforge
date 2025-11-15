import numpy as np
from ..core import X, Y, Z

"""
This module provides helper functions for achieving CAD-style geometric
relationships through construction rather than through a constraint solver.

In a traditional CAD system, a solver adjusts object properties to meet
constraints. In `sdforge`, we achieve the same results by explicitly calculating
the required transformations (positions, rotations) before creating the final
object. These functions encapsulate that geometric math.
"""

def coincident(point_a, point_b):
    """
    Checks if two points are coincident within a small tolerance.
    This is primarily for validation or testing, as the best practice
    is to simply use the same variable for both points.

    Args:
        point_a (tuple or np.ndarray): The first point.
        point_b (tuple or np.ndarray): The second point.

    Returns:
        bool: True if the points are effectively coincident.
    """
    return np.allclose(point_a, point_b, atol=1e-6)

def midpoint(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """
    Calculates the midpoint between two points.

    Args:
        point_a (np.ndarray): The first point.
        point_b (np.ndarray): The second point.

    Returns:
        np.ndarray: The point halfway between point_a and point_b.
    """
    return (np.array(point_a) + np.array(point_b)) / 2.0


def tangent_offset(circle_radius: float, line_direction: np.ndarray) -> np.ndarray:
    """
    Calculates the translational offset required to make a line, passing
    through the origin, tangent to a circle centered at the origin.

    The offset is perpendicular to the line's direction.

    Args:
        circle_radius (float): The radius of the circle.
        line_direction (np.ndarray): The normalized direction vector of the line.

    Returns:
        np.ndarray: The 3D vector offset to apply to the line's translation.
    """
    # Find a vector perpendicular to the line direction in the XY plane
    # This is a simplification; a robust solution would handle 3D cases.
    perp_vec = np.array([-line_direction[1], line_direction[0], 0.0])
    return perp_vec * circle_radius

def align_to_face(obj_to_align, reference_point, face_normal: np.ndarray, offset: float = 0.0):
    """
    Applies transformations to an object to align it with a conceptual
    "face" defined by a point and a normal vector.

    For example, align a cylinder to be perpendicular to the +X face of a cube
    by providing a point on that face and the face's normal vector.

    Args:
        obj_to_align (SDFNode): The object to move and rotate.
        reference_point (np.ndarray): The point on the conceptual "face" to
                                      which the object will be aligned.
        face_normal (np.ndarray): The normal vector of the face to align to
                                  (e.g., sdforge.X for the +X face).
        offset (float, optional): An additional distance to move the object
                                  along the face normal. Defaults to 0.0.

    Returns:
        SDFNode: The transformed (aligned and translated) object.
    """
    
    # Normalize the face normal to be safe
    face_normal = np.array(face_normal) / np.linalg.norm(face_normal)
    
    # --- Rotation ---
    # The default orientation of most shapes (like cylinder) is along the Y-axis.
    # We need to find the rotation that takes the Y-axis to the face_normal.
    y_axis = Y
    
    # If the vectors are already aligned (or opposite), no rotation is needed in that plane
    if np.allclose(y_axis, face_normal):
        rotated_obj = obj_to_align
    elif np.allclose(y_axis, -face_normal):
        # Rotate 180 degrees around X-axis (or any perpendicular axis)
        rotated_obj = obj_to_align.rotate(X, np.pi)
    else:
        # This simple axis-angle rotation can be tricky to implement with
        # SDForge's cardinal-axis-only .rotate(). For this helper, we'll
        # simplify by only supporting cardinal face normals.
        
        if np.allclose(np.abs(face_normal), X):
            # Rotate from Y to X or -X
            rotated_obj = obj_to_align.rotate(Z, -np.sign(face_normal[0]) * np.pi / 2)
        elif np.allclose(np.abs(face_normal), Y):
            # Already aligned, or needs 180 deg spin
            rotated_obj = obj_to_align if face_normal[1] > 0 else obj_to_align.rotate(X, np.pi)
        elif np.allclose(np.abs(face_normal), Z):
            # Rotate from Y to Z or -Z
            rotated_obj = obj_to_align.rotate(X, np.sign(face_normal[2]) * np.pi / 2)
        else:
            raise ValueError("align_to_face currently only supports cardinal axis normals (X, Y, Z).")

    # --- Translation ---
    # Estimate the half-size of the object to be aligned. This is a heuristic.
    # A robust implementation would require bounding box info from the object.
    # For a cylinder, its height/2 is its primary dimension.
    try:
        obj_half_size = obj_to_align.height / 2.0
    except AttributeError:
        obj_half_size = 0.5 # Default fallback

    # The object is placed such that its base is centered on the reference_point,
    # then pushed out along the normal by its own half-size and any extra offset.
    translation = reference_point + face_normal * (obj_half_size + offset)
    
    return rotated_obj.translate(translation)

def place_at_angle(obj_to_place, pivot_point, axis, angle_rad, distance):
    """
    Places an object at a specific angle and distance from a pivot point.

    This is useful for creating circular patterns of features, like holes on a flange.
    The function performs the following steps:
    1. Translates the object out from the origin by `distance` along the X-axis.
    2. Rotates the object around the specified `axis` by `angle_rad`.
    3. Translates the rotated object to the `pivot_point`.

    Args:
        obj_to_place (SDFNode): The feature to place.
        pivot_point (np.ndarray): The center of the angular pattern.
        axis (np.ndarray): The axis of rotation for the angle (e.g., sdforge.Y).
        angle_rad (float): The angle in radians.
        distance (float): The radial distance from the pivot point.

    Returns:
        SDFNode: The transformed object.
    """
    return obj_to_place.translate(X * distance).rotate(axis, angle_rad).translate(pivot_point)

def offset_along(obj_to_place, reference_point, direction, distance):
    """
    Translates an object to a new position offset from a reference point
    along a specific direction vector.

    Args:
        obj_to_place (SDFNode): The object to move.
        reference_point (np.ndarray): The starting point for the offset.
        direction (np.ndarray): The normalized vector for the direction of offset.
        distance (float): The distance to move along the direction vector.

    Returns:
        SDFNode: The translated object.
    """
    normalized_dir = np.array(direction) / np.linalg.norm(direction)
    destination = np.array(reference_point) + normalized_dir * distance
    return obj_to_place.translate(destination)

def bounding_box(sdf_obj, padding: float = 0.0):
    """
    Computes the bounding box of a complex SDF object and returns it as a
    simple `box` primitive.

    This is a powerful tool for creating enclosures, fixtures, or performing
    boolean operations on the entire volume of another part. It works by
    sampling the SDF to find its extents.

    Args:
        sdf_obj (SDFNode): The object to measure.
        padding (float, optional): An extra border to add to all sides of
                                   the bounding box. Defaults to 0.0.

    Returns:
        SDFNode: A `box` primitive that encloses the original object.
    """
    from ..api.primitives import box # Local import to avoid circular dependency
    bounds = sdf_obj.estimate_bounds(verbose=False)
    min_c, max_c = np.array(bounds[0]), np.array(bounds[1])
    
    size = max_c - min_c + (2 * padding)
    center = (min_c + max_c) / 2.0
    
    return box(size=tuple(size)).translate(tuple(center))