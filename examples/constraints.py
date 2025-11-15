import sys
import numpy as np
from sdforge import box, sphere, cylinder, line, Param, Group, X, Y, Z
from sdforge.api.constraints import (
    tangent_offset,
    align_to_face,
    midpoint,
    place_at_angle,
    offset_along,
    bounding_box,
)

def dimensional_equality_example():
    """
    Demonstrates dimensional equality by using the same Param object
    for multiple dimensions, ensuring they always match.
    """
    # This parameter will control both the hole size and the sphere size
    p_hole_dia = Param("Hole Diameter", 0.8, 0.2, 1.5)
    
    # The main block
    block = box(size=(3, 1, 2))
    
    # A hole whose radius is driven by the parameter
    hole = cylinder(radius=p_hole_dia / 2, height=1.1)
    
    # A sphere whose radius is also driven by the same parameter
    sphere_on_top = sphere(r=p_hole_dia / 2).translate((0, 1.0, 0))
    
    scene = (block - hole) | sphere_on_top
    return scene

def tangent_and_coincident_example():
    """
    Demonstrates creating tangent and coincident relationships through
    explicit calculation and construction.
    """
    # Define a center point. Any object using this variable will be
    # 'coincident' at this point. Use floats to prevent casting errors.
    center_point = np.array([0.0, 0.0, 0.0])
    
    # A central circle
    c = cylinder(radius=1.0, height=0.2)
    
    # We want a line tangent to the circle. First, define its direction.
    # Use floats to prevent casting errors during normalization.
    line_dir = np.array([1.0, 1.0, 0.0])
    line_dir /= np.linalg.norm(line_dir) # Normalize
    
    # Use the helper to calculate the translation needed for tangency
    offset = tangent_offset(circle_radius=1.0, line_direction=line_dir)
    
    # The line is constructed with the correct offset.
    # The start/end points use the 'center_point' and 'offset' variables,
    # ensuring the relationship is maintained if they change.
    tangent_line = line(
        a = center_point + offset - line_dir * 2,
        b = center_point + offset + line_dir * 2,
        radius=0.1
    )
    
    scene = c | tangent_line
    return scene

def perpendicular_and_parallel_example():
    """
    Demonstrates creating perpendicular and parallel features by construction,
    using the align_to_face helper.
    """
    # Main body
    main_block = box(size=(2.0, 1.0, 1.5))
    main_block_half_size = np.array(main_block.size) / 2.0

    # A boss to be placed on the +X face, perpendicular to it.
    boss = cylinder(radius=0.3, height=0.5)
    
    # Define the point on the center of the +X face of the block
    face_point_positive_x = X * main_block_half_size[0]

    # Use the helper to calculate rotation and translation.
    # The boss's axis (Y) will be aligned with the face normal (X),
    # making it perpendicular to the face. It will be placed on the face_point.
    aligned_boss = align_to_face(boss, reference_point=face_point_positive_x, face_normal=X)

    # A hole through the -X face.
    hole = cylinder(radius=0.2, height=2.1)
    face_point_negative_x = -X * main_block_half_size[0]
    aligned_hole = align_to_face(hole, reference_point=face_point_negative_x, face_normal=-X)
    
    # The boss and hole are now parallel to each other because they were
    # both constructed to be parallel to the X-axis.
    
    scene = (main_block | aligned_boss) - aligned_hole
    return scene

def angle_and_midpoint_example():
    """
    Demonstrates placing features at an angle and at the midpoint of an edge.
    """
    # A flange-like plate
    flange = cylinder(radius=2.0, height=0.5) - cylinder(radius=1.5, height=0.6)
    
    # A single hole feature to be patterned
    hole = cylinder(radius=0.2, height=0.6)
    
    # Use the place_at_angle helper to create a circular pattern of 6 holes
    num_holes = 6
    holes = []
    for i in range(num_holes):
        angle = i * (2 * np.pi / num_holes)
        h = place_at_angle(
            obj_to_place=hole, 
            pivot_point=(0,0,0), 
            axis=Y, 
            angle_rad=angle, 
            distance=1.75
        )
        holes.append(h)
    
    # A slot feature placed at the midpoint of a conceptual line
    corner_a = np.array([2.5, 0, -0.5])
    corner_b = np.array([2.5, 0, 0.5])
    slot_center = midpoint(corner_a, corner_b)
    slot = box(size=(0.2, 0.6, 1.5), radius=0.1).translate(slot_center)
    
    scene = (flange - Group(*holes)) | slot
    return scene

def offset_and_bounds_example():
    """
    Demonstrates offsetting an object along a vector and creating a
    bounding box around a complex shape.
    """
    # A complex, non-symmetrical base part
    base_part = box(1.0).scale((2, 0.5, 1)) | sphere(0.8).translate(X * 0.7)
    
    # Use offset_along to place a support pillar relative to the base part.
    # We offset from the origin along a diagonal vector.
    pillar_start_point = np.array([0.0, -1.0, 0.0])
    pillar_direction = np.array([-1.0, -0.5, 1.0])
    pillar = cylinder(0.2, 1.5)
    placed_pillar = offset_along(pillar, pillar_start_point, pillar_direction, 2.0)
    
    # Generate a bounding box around the complex base part to create a shell.
    # The bounding box itself is just another SDF primitive.
    bbox = bounding_box(base_part, padding=0.1)
    
    # Make the bounding box hollow to create an enclosure
    enclosure = bbox.shell(0.05)
    
    # We display the base part (in red) and the pillar inside the enclosure.
    final_scene = enclosure.color(0.4, 0.4, 0.5) | base_part.color(1, 0, 0) | placed_pillar
    return final_scene


def design_patterns_example():
    """
    Demonstrates achieving constraints like concentricity and symmetry
    using core SDForge features as a design pattern.
    """
    # --- Concentricity ---
    # By defining a single center point and translating multiple objects by
    # that same vector, they are guaranteed to be concentric.
    center = np.array([2.0, 0.5, 0.0])
    
    outer_cyl = cylinder(radius=0.8, height=1.0).translate(center)
    inner_cyl = cylinder(radius=0.5, height=1.2).translate(center)
    concentric_part = outer_cyl - inner_cyl
    
    # --- Symmetry ---
    # The mirror() operation is the primary tool for creating symmetry.
    # We model one quadrant of a complex shape...
    quarter_shape = box(size=(1, 0.4, 1), radius=0.1).translate((0.5, 0.2, 0.5))
    cutout = sphere(0.3).translate((0.8, 0.2, 0.8))
    quarter_part = quarter_shape - cutout
    
    # ...then mirror it across the X and Z axes to create the full part.
    # The final object is guaranteed to be symmetrical.
    symmetrical_part = quarter_part.mirror(X | Z).translate((-2.0, 0, 0))
    
    scene = concentric_part | symmetrical_part
    return scene

def main():
    """
    Renders a constraints example based on a command-line argument.
    """
    print("--- SDForge Constraints by Construction Examples ---")
    
    examples = {
        "dims_equal": dimensional_equality_example,
        "tangent": tangent_and_coincident_example,
        "alignment": perpendicular_and_parallel_example,
        "angle_midpoint": angle_and_midpoint_example,
        "offset_bounds": offset_and_bounds_example,
        "patterns": design_patterns_example,
    }
    
    if len(sys.argv) < 2:
        print("\nPlease provide the name of an example to run.")
        print("Available examples:")
        for key in examples:
            print(f"  - {key}")
        print(f"\nUsage: python {sys.argv[0]} <example_name>")
        return

    example_name = sys.argv[1]
    scene_func = examples.get(example_name)
    
    if not scene_func:
        print(f"\nError: Example '{example_name}' not found.")
        return

    print(f"Rendering: {example_name.replace('_', ' ').title()} Example")
    scene = scene_func()
    scene.render()

if __name__ == "__main__":
    main()