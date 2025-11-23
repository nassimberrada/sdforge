import sys
import numpy as np
from sdforge import sphere, box, cylinder, X, Param

def union_example():
    """A sphere and a box joined together."""
    s = sphere(radius=0.8)
    b = box(size=(1.5, 0.5, 0.5))
    return s | b

def intersection_example():
    """A lens shape created by intersecting two spheres."""
    s1 = sphere(radius=1.0).translate((-0.5, 0, 0))
    s2 = sphere(radius=1.0).translate((0.5, 0, 0))
    return s1 & s2

def difference_example():
    """A box with a sphere carved out of it."""
    b = box(size=1.5)
    s = sphere(radius=1.0)
    return b - s

def smooth_union_example():
    """Two spheres smoothly blended together."""
    s1 = sphere(radius=0.7).translate((-0.5, 0, 0))
    s2 = sphere(radius=0.7).translate((0.5, 0, 0))
    # The 'blend' parameter controls the smoothness of the blend.
    return s1.union(s2, blend=0.3)

def fillet_difference_example():
    """Demonstrates subtracting a shape with a rounded (filleted) edge."""
    plate = box(size=(2.0, 0.5, 2.0))
    hole = cylinder(radius=0.5, height=1.0)

    # blend=0.1 creates a rounded edge where the cylinder is subtracted.
    scene = plate.difference(hole, blend=0.1)
    return scene

def linear_difference_example():
    """Demonstrates subtracting a shape with a linear (chamfered) edge."""
    plate = box(size=(2.0, 0.5, 2.0))
    hole = cylinder(radius=0.5, height=1.0)

    # blend=0.1, blend_type='linear' creates a 45-degree bevel.
    scene = plate.difference(hole, blend=0.1, blend_type='linear')
    return scene

def fillet_union_example():
    """Demonstrates joining two shapes with a filleted seam."""
    s1 = sphere(0.8).translate(-X * 0.5)
    s2 = sphere(0.8).translate(X * 0.5)

    # The blend parameter smoothly blends the two spheres together.
    scene = s1.union(s2, blend=0.4)
    return scene

def morphing_example():
    """
    Demonstrates interpolating between two shapes (morphing).
    Uses a Param to make it interactive.
    """
    s = sphere(radius=1.0)
    b = box(size=1.5)

    # Use a parameter to control the morph factor (0.0 = sphere, 1.0 = box)
    p_morph = Param("Morph Factor", 0.5, 0.0, 1.0)

    return s.morph(b, factor=p_morph)

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Operation Examples ---")

    examples = {
        "union": union_example,
        "intersection": intersection_example,
        "difference": difference_example,
        "smooth_union": smooth_union_example,
        "fillet_cut": fillet_difference_example,
        "linear_cut": linear_difference_example,
        "fillet_join": fillet_union_example,
        "morph": morphing_example,
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
        print("Available examples are:")
        for key in examples:
            print(f"  - {key}")
        return

    print(f"Rendering: {example_name.replace('_', ' ').title()} Example")
    result = scene_func()
    if isinstance(result, tuple):
        scene, cam = result
        scene.render(camera=cam)
    else:
        scene = result
        scene.render()


if __name__ == "__main__":
    main()