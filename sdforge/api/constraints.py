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

    Args:
        point_a (tuple or np.ndarray): The first point.
        point_b (tuple or np.ndarray): The second point.
    
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
    
    Example:
        >>> corner1 = np.array([-1, -1, -1])
        >>> corner2 = np.array([1, 1, 1])
        >>> center = midpoint(corner1, corner2) # returns array([0., 0., 0.])
        >>> b = box(0.5).translate(center)
    """
    return (np.array(point_a) + np.array(point_b)) / 2.0


def tangent_offset(circle_radius: float, line_direction: np.ndarray) -> np.ndarray:
    """
    Calculates the offset to make a line tangent to a circle.

    Args:
        circle_radius (float): The radius of the circle.
        line_direction (np.ndarray): The normalized direction vector of the line.
    
    Example:
        >>> circle_obj = cylinder(radius=2.0, height=0.1)
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

    Args:
        obj_to_align (SDFNode): The object to move and rotate.
        reference_point (np.ndarray): The point on the conceptual "face" to
                                      which the object will be aligned.
        face_normal (np.ndarray): The normal vector of the face to align to.
        offset (float, optional): An additional distance to move the object
                                  along the face normal. Defaults to 0.0.
    
    Example:
        >>> main_box = box(size=(2, 1, 1))
        >>> face_center = (1, 0, 0)
        >>> boss = cylinder(radius=0.2, height=0.5)
        >>> placed_boss = align_to_face(boss, face_center, face_normal=X)
        >>> scene = main_box | placed_boss
    """
    
    # Normalize the face normal to be safe
    face_normal = np.array(face_normal) / np.linalg.norm(face_normal)
    
    # --- Rotation ---
    y_axis = Y
    
    if np.allclose(y_axis, face_normal):
        rotated_obj = obj_to_align
    elif np.allclose(y_axis, -face_normal):
        rotated_obj = obj_to_align.rotate(X, np.pi)
    else:
        if np.allclose(np.abs(face_normal), X):
            rotated_obj = obj_to_align.rotate(Z, -np.sign(face_normal[0]) * np.pi / 2)
        elif np.allclose(np.abs(face_normal), Y):
            rotated_obj = obj_to_align if face_normal[1] > 0 else obj_to_align.rotate(X, np.pi)
        elif np.allclose(np.abs(face_normal), Z):
            rotated_obj = obj_to_align.rotate(X, np.sign(face_normal[2]) * np.pi / 2)
        else:
            raise ValueError("align_to_face currently only supports cardinal axis normals (X, Y, Z).")

    # --- Translation ---
    try:
        obj_half_size = obj_to_align.height / 2.0
    except AttributeError:
        obj_half_size = 0.5 

    translation = reference_point + face_normal * (obj_half_size + offset)
    
    return rotated_obj.translate(translation)

def place_at_angle(obj_to_place, pivot_point, axis, angle_rad, distance):
    """
    Places an object at a specific angle and distance from a pivot point.

    Args:
        obj_to_place (SDFNode): The feature to place.
        pivot_point (np.ndarray): The center of the angular pattern.
        axis (np.ndarray): The axis of rotation for the angle.
        angle_rad (float): The angle in radians.
        distance (float): The radial distance from the pivot point.
    
    Example:
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

    Args:
        obj_to_place (SDFNode): The object to move.
        reference_point (np.ndarray): The starting point for the offset.
        direction (np.ndarray): The vector for the direction of offset.
        distance (float): The distance to move along the direction vector.
        
    Example:
        >>> start = (1, 1, 1)
        >>> direction = (1, 0, 0)
        >>> s = sphere(0.5)
        >>> placed_sphere = offset_along(s, start, direction, 5.0)
    """
    normalized_dir = np.array(direction) / np.linalg.norm(direction)
    destination = np.array(reference_point) + normalized_dir * distance
    return obj_to_place.translate(destination)

def bounding_box(sdf_obj, padding: float = 0.0):
    """
    Creates a `box` primitive that encloses a complex SDF object.

    Args:
        sdf_obj (SDFNode): The object to measure.
        padding (float, optional): An extra border to add to all sides.

    Example:
        >>> complex_shape = sphere(1.0) | sphere(0.5).translate(X * 1.5)
        >>> enclosure = bounding_box(complex_shape, padding=0.1).shell(0.05)
        >>> scene = enclosure | complex_shape.color(1,0,0)
    """
    from ..api.primitives import box # Local import to avoid circular dependency
    bounds = sdf_obj.estimate_bounds(verbose=False)
    min_c, max_c = np.array(bounds[0]), np.array(bounds[1])
    
    size = max_c - min_c + (2 * padding)
    center = (min_c + max_c) / 2.0
    
    return box(size=tuple(size)).translate(tuple(center))

def compute_stack_transform(obj_fixed, obj_movable, direction, spacing=0.0):
    """
    Calculates the translation vector required to stack obj_movable onto obj_fixed.
    Used internally by stack() and distribute().
    """
    direction = np.array(direction, dtype=float)
    len_dir = np.linalg.norm(direction)
    if len_dir == 0: raise ValueError("Direction cannot be zero.")
    direction /= len_dir
    
    # 1. Estimate bounds
    b_fixed = obj_fixed.estimate_bounds(verbose=False)
    b_movable = obj_movable.estimate_bounds(verbose=False)
    
    min_f, max_f = np.array(b_fixed[0]), np.array(b_fixed[1])
    min_m, max_m = np.array(b_movable[0]), np.array(b_movable[1])
    
    # 2. Centers
    c_f = (min_f + max_f) / 2.0
    c_m = (min_m + max_m) / 2.0
    
    # 3. Initial Alignment: Match centers
    T = c_f - c_m
    
    # 4. Project bounds onto dominant axis
    # This logic assumes reasonably axis-aligned usage, which is standard for constructive layouts.
    abs_dir = np.abs(direction)
    axis_idx = np.argmax(abs_dir)
    sign = np.sign(direction[axis_idx])
    
    # Determine the "face" coordinate on the fixed object
    # If direction is +Y, we want the Max Y of fixed.
    fixed_face = max_f[axis_idx] if sign > 0 else min_f[axis_idx]
    
    # Determine the "face" coordinate on the movable object
    # If direction is +Y, we want the Min Y of movable to touch fixed.
    movable_face = min_m[axis_idx] if sign > 0 else max_m[axis_idx]
    
    # 5. Calculate shift along the axis
    # The movable object is currently conceptually at its original position + T (centered).
    # We need to adjust T[axis_idx] so that:
    #   (movable_face relative to old center) + new_center = fixed_face + spacing
    
    # Current aligned position of the movable face
    # movable_face is absolute coord. 
    # The centering transform T moves everything by (c_f - c_m).
    aligned_movable_face_pos = movable_face + T[axis_idx]
    
    target_pos = fixed_face + (sign * spacing)
    
    diff = target_pos - aligned_movable_face_pos
    T[axis_idx] += diff
    
    return T

def stack(obj_fixed, obj_movable, direction, spacing=0.0):
    """
    Stacks `obj_movable` onto `obj_fixed` along the given direction.

    Args:
        obj_fixed (SDFNode): The stationary base object.
        obj_movable (SDFNode): The object to move and stack.
        direction (tuple): Vector direction to stack (e.g., (0, 1, 0) for on top).
        spacing (float, optional): Gap between the objects. Defaults to 0.0.
    
    Example:
        >>> b = box(2.0)
        >>> s = sphere(1.0)
        >>> scene = stack(b, s, direction=(0, 1, 0))
    """
    T = compute_stack_transform(obj_fixed, obj_movable, direction, spacing)
    return obj_fixed | obj_movable.translate(T)

def distribute(objects, direction, spacing=0.0):
    """
    Arranges a list of objects sequentially along a direction.
    
    Args:
        objects (list): List of SDFNodes to distribute.
        direction (tuple): Direction vector for the layout.
        spacing (float, optional): Gap between adjacent objects. Defaults to 0.0.
    """
    if not objects:
        return None
    
    from ..api.group import Group
    
    transformed_list = [objects[0]]
    
    for i in range(1, len(objects)):
        prev_obj = transformed_list[-1]
        curr_obj = objects[i]
        
        T = compute_stack_transform(prev_obj, curr_obj, direction, spacing)
        transformed_list.append(curr_obj.translate(T))
        
    return Group(*transformed_list)