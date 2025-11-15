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
    is to simply use the same variable for both points to enforce coincidence
    by construction.

    Args:
        point_a (tuple or np.ndarray): The first point.
        point_b (tuple or np.ndarray): The second point.

    Returns:
        bool: True if the points are effectively coincident.
    
    Example:
        >>> p1 = (1.0, 2.0, 3.0)
        >>> p2 = (1.00000001, 2.0, 3.0)
        >>> coincident(p1, p2)
        True
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
    
    Example:
        >>> from sdforge import box
        >>> import numpy as np
        >>> corner1 = np.array([-1, -1, -1])
        >>> corner2 = np.array([1, 1, 1])
        >>> center = midpoint(corner1, corner2) # returns array([0., 0., 0.])
        >>> # Create a box centered between two points
        >>> b = box(0.5).translate(center)
    """
    return (np.array(point_a) + np.array(point_b)) / 2.0


def tangent_offset(circle_radius: float, line_direction: np.ndarray) -> np.ndarray:
    """
    Calculates the offset to make a line tangent to a circle.

    This helper computes the translational offset required to make a line,
    passing through the origin, tangent to a circle also centered at the origin.
    The offset is perpendicular to the line's direction.

    Args:
        circle_radius (float): The radius of the circle.
        line_direction (np.ndarray): The normalized direction vector of the line.

    Returns:
        np.ndarray: The 3D vector offset to apply to the line's translation.
    
    Example:
        >>> from sdforge import line, cylinder, X
        >>> circle_obj = cylinder(radius=2.0, height=0.1)
        >>> # Make a line tangent to the circle, along the X axis
        >>> offset = tangent_offset(circle_radius=2.0, line_direction=X)
        >>> tangent = line(a=(-5,0,0), b=(5,0,0), radius=0.05).translate(offset)
        >>> scene = circle_obj | tangent
    """
    # Find a vector perpendicular to the line direction in the XY plane
    # This is a simplification; a robust solution would handle 3D cases.
    perp_vec = np.array([-line_direction[1], line_direction[0], 0.0])
    return perp_vec * circle_radius

def align_to_face(obj_to_align, reference_point, face_normal: np.ndarray, offset: float = 0.0):
    """
    Aligns and places an object onto a conceptual face.

    This function applies transformations to an object to align its primary
    axis (usually Y) with a "face" defined by a point and a normal vector.
    It's useful for placing features like bosses or holes onto the surface
    of another object. Note: currently only supports cardinal axis normals.

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
    
    Example:
        >>> from sdforge import box, cylinder, X
        >>> main_box = box(size=(2, 1, 1))
        >>> # Place a cylinder on the center of the +X face of the box
        >>> face_center = (1, 0, 0)
        >>> boss = cylinder(radius=0.2, height=0.5)
        >>> placed_boss = align_to_face(boss, face_center, face_normal=X)
        >>> scene = main_box | placed_boss
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
    The function performs a sequence of translate-rotate-translate operations.

    Args:
        obj_to_place (SDFNode): The feature to place.
        pivot_point (np.ndarray): The center of the angular pattern.
        axis (np.ndarray): The axis of rotation for the angle (e.g., sdforge.Y).
        angle_rad (float): The angle in radians.
        distance (float): The radial distance from the pivot point.

    Returns:
        SDFNode: The transformed object.
    
    Example:
        >>> import numpy as np
        >>> from sdforge import cylinder, Group, Y
        >>> flange = cylinder(radius=2.0, height=0.2)
        >>> hole = cylinder(radius=0.1, height=0.3)
        >>> holes = []
        >>> for i in range(6):
        ...     angle = i * (2 * np.pi / 6)
        ...     h = place_at_angle(hole, (0,0,0), Y, angle, 1.75)
        ...     holes.append(h)
        >>> scene = flange - Group(*holes)
    """
    return obj_to_place.translate(X * distance).rotate(axis, angle_rad).translate(pivot_point)

def offset_along(obj_to_place, reference_point, direction, distance):
    """
    Moves an object from a reference point along a direction vector.

    Translates an object to a new position that is a specific distance
    away from a reference point along a given direction.

    Args:
        obj_to_place (SDFNode): The object to move.
        reference_point (np.ndarray): The starting point for the offset.
        direction (np.ndarray): The vector for the direction of offset. It will be normalized.
        distance (float): The distance to move along the direction vector.

    Returns:
        SDFNode: The translated object.
        
    Example:
        >>> from sdforge import sphere
        >>> start = (1, 1, 1)
        >>> direction = (1, 0, 0)
        >>> # Place a sphere 5 units along the X axis from the start point
        >>> s = sphere(0.5)
        >>> placed_sphere = offset_along(s, start, direction, 5.0)
        >>> # Final position will be (6, 1, 1)
    """
    normalized_dir = np.array(direction) / np.linalg.norm(direction)
    destination = np.array(reference_point) + normalized_dir * distance
    return obj_to_place.translate(destination)

def bounding_box(sdf_obj, padding: float = 0.0):
    """
    Creates a `box` primitive that encloses a complex SDF object.

    This powerful tool works by sampling the SDF to find its extents, then
    returning a simple `box` primitive that matches those bounds. It's useful
    for creating enclosures, fixtures, or performing boolean operations on
    the entire volume of another part.

    Args:
        sdf_obj (SDFNode): The object to measure.
        padding (float, optional): An extra border to add to all sides of
                                   the bounding box. Defaults to 0.0.

    Returns:
        SDFNode: A `box` primitive that encloses the original object.
    
    Example:
        >>> from sdforge import sphere, X
        >>> # A complex, off-center object
        >>> complex_shape = sphere(1.0) | sphere(0.5).translate(X * 1.5)
        >>> # Create a shell around it using its bounding box
        >>> enclosure = bounding_box(complex_shape, padding=0.1).shell(0.05)
        >>> scene = enclosure | complex_shape.color(1,0,0)
    """
    from ..api.primitives import box # Local import to avoid circular dependency
    bounds = sdf_obj.estimate_bounds(verbose=False)
    min_c, max_c = np.array(bounds[0]), np.array(bounds[1])
    
    size = max_c - min_c + (2 * padding)
    center = (min_c + max_c) / 2.0
    
    return box(size=tuple(size)).translate(tuple(center))