import sys
import numpy as np
from sdforge import box, sphere, cylinder, line, Param, Group, X, Y, Z, cone
from sdforge.api.constraints import (
    tangent_offset,
    midpoint,
    distribute,
)

def dimensional_equality_example():
    """
    Demonstrates dimensional equality by using the same Param object
    for multiple dimensions, ensuring they always match.
    """
    p_hole_dia = Param("Hole Diameter", 0.8, 0.2, 1.5)
    block = box(size=(3, 1, 2))
    hole = cylinder(radius=p_hole_dia / 2, height=1.1)
    sphere_on_top = sphere(radius=p_hole_dia / 2).translate((0, 1.0, 0))

    scene = (block - hole) | sphere_on_top
    return scene

def tangent_and_coincident_example():
    """
    Demonstrates creating tangent and coincident relationships through
    explicit calculation and construction.
    """
    center_point = np.array([0.0, 0.0, 0.0])
    c = cylinder(radius=1.0, height=0.2)

    line_dir = np.array([1.0, 1.0, 0.0])
    line_dir /= np.linalg.norm(line_dir) 

    offset = tangent_offset(circle_radius=1.0, line_direction=line_dir)

    tangent_line = line(
        start = center_point + offset - line_dir * 2,
        end = center_point + offset + line_dir * 2,
        radius=0.1
    )

    scene = c | tangent_line
    return scene

def perpendicular_and_parallel_example():
    """
    Demonstrates creating perpendicular and parallel features by construction,
    using the .align_to() method.
    """
    main_block = box(size=(2.0, 1.0, 1.5))
    main_block_half_size = np.array(main_block.size) / 2.0

    # A boss to be placed on the +X face, perpendicular to it.
    boss = cylinder(radius=0.3, height=0.5)

    # Define the point on the center of the +X face of the block
    face_point_positive_x = X * main_block_half_size[0]

    # Use the fluent helper method .align_to()
    aligned_boss = boss.align_to(reference_point=face_point_positive_x, face_normal=X)

    # A hole through the -X face.
    hole = cylinder(radius=0.2, height=2.1)
    face_point_negative_x = -X * main_block_half_size[0]
    aligned_hole = hole.align_to(reference_point=face_point_negative_x, face_normal=-X)

    scene = (main_block | aligned_boss) - aligned_hole
    return scene

def angle_and_midpoint_example():
    """
    Demonstrates placing features at an angle and at the midpoint of an edge.
    """
    flange = cylinder(radius=2.0, height=0.5) - cylinder(radius=1.5, height=0.6)
    hole = cylinder(radius=0.2, height=0.6)

    # Use the fluent .place_at_angle() method
    num_holes = 6
    holes = []
    for i in range(num_holes):
        angle = i * (2 * np.pi / num_holes)
        h = hole.place_at_angle(
            pivot_point=(0,0,0), 
            axis=Y, 
            angle_rad=angle, 
            distance=1.75
        )
        holes.append(h)

    corner_a = np.array([2.5, 0, -0.5])
    corner_b = np.array([2.5, 0, 0.5])
    slot_center = midpoint(corner_a, corner_b)
    slot = box(size=(0.2, 0.6, 1.5)).round(0.1).translate(slot_center)

    scene = (flange - Group(*holes)) | slot
    return scene

def offset_and_bounds_example():
    """
    Demonstrates offsetting an object along a vector and creating a
    bounding box around a complex shape.
    """
    base_part = box(1.0).scale((2, 0.5, 1)) | sphere(radius=0.8).translate(X * 0.7)

    # Use fluent .offset_along()
    pillar_start_point = np.array([0.0, -1.0, 0.0])
    pillar_direction = np.array([-1.0, -0.5, 1.0])
    pillar = cylinder(0.2, 1.5)
    placed_pillar = pillar.offset_along(pillar_start_point, pillar_direction, 2.0)

    # Generate a bounding box using the fluent .bounding_box() method
    bbox = base_part.bounding_box(padding=0.1)

    # Make the bounding box hollow
    enclosure = bbox.shell(0.05)

    final_scene = enclosure.color(0.4, 0.4, 0.5) | base_part.color(1, 0, 0) | placed_pillar
    return final_scene

def stacking_example():
    """
    Demonstrates the .stack() method to place objects relative to each other
    without manual coordinate calculations.
    """
    # 1. Start with a large base
    base = box(size=(3.0, 0.5, 3.0)).round(0.1)
    
    # 2. Stack a cylinder on top (+Y)
    # The system calculates the top face of the box and bottom face of the cylinder
    # and positions them to touch.
    pedestal = cylinder(radius=1.0, height=1.0)
    stage_1 = base.stack(pedestal, direction=(0, 1, 0))
    
    # 3. Stack a sphere on top of the resulting union, with a small gap
    orb = sphere(radius=0.8).color(1, 0.2, 0.2)
    final_scene = stage_1.stack(orb, direction=(0, 1, 0), spacing=0.2)
    
    return final_scene

def distribute_example():
    """
    Demonstrates the distribute() function to layout multiple objects sequentially.
    """
    shapes = [
        box(1.0).color(0.8, 0.2, 0.2),       # Red Box
        sphere(0.6).color(0.2, 0.8, 0.2),    # Green Sphere
        cylinder(0.4, 1.5).color(0.2, 0.2, 0.8), # Blue Cylinder
        cone(1.0, 0.5).color(0.8, 0.8, 0.2)  # Yellow Cone
    ]
    
    # Distribute them along the X axis with a 0.5 unit gap between each
    scene = distribute(shapes, direction=(1, 0, 0), spacing=0.5)
    
    return scene

def design_patterns_example():
    """
    Demonstrates achieving constraints like concentricity and symmetry
    using core SDForge features as a design pattern.
    """
    center = np.array([2.0, 0.5, 0.0])

    outer_cyl = cylinder(radius=0.8, height=1.0).translate(center)
    inner_cyl = cylinder(radius=0.5, height=1.2).translate(center)
    concentric_part = outer_cyl - inner_cyl

    quarter_shape = box(size=(1, 0.4, 1)).round(0.1).translate((0.5, 0.2, 0.5))
    cutout = sphere(radius=0.3).translate((0.8, 0.2, 0.8))
    quarter_part = quarter_shape - cutout

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
        "stack": stacking_example,
        "distribute": distribute_example,
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