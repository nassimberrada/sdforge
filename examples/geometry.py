import sys
from sdforge import (
    sphere, box, cylinder, torus, cone, 
    hex_prism, pyramid, curve,
    circle, rectangle, triangle, trapezoid, polyline, polycurve,
    X, Y, Z
)

def sphere_example():
    """Returns a simple sphere scene."""
    s = sphere(radius=1.0)
    return s

def box_example():
    """Returns a simple box scene."""
    return box(size=(1.5, 1.0, 0.5)).round(0.1)

def cylinder_example():
    """Returns a simple cylinder scene."""
    return cylinder(radius=0.5, height=1.5)

def torus_example():
    """Returns a simple torus scene."""
    return torus(radius_major=1.0, radius_minor=0.25)

def cone_example():
    """Returns a frustum (capped cone) scene."""
    return cone(height=1.2, radius_base=0.6, radius_top=0.2)

def hex_prism_example():
    """Returns a hexagonal prism."""
    return hex_prism(radius=1.0, height=0.5)

def pyramid_example():
    """Returns a pyramid."""
    return pyramid(height=1.2)

def curve_example():
    """Returns a curved tube (Quadratic Bezier)."""
    return curve(
        p0=(-1.0, 0.0, 0.0), 
        p1=(0.0, 1.5, 0.0), 
        p2=(1.0, 0.0, 0.0), 
        radius=0.1
    )

def circle_example():
    """Returns a flat circle (disc)."""
    return circle(radius=1.0)

def rectangle_example():
    """Returns a flat rectangle (plate)."""
    return rectangle(size=(1.5, 1.0))

def triangle_example():
    """Returns a flat equilateral triangle."""
    return triangle(radius=1.0)

def trapezoid_example():
    """Returns a flat isosceles trapezoid."""
    return trapezoid(bottom_width=1.5, top_width=0.8, height=1.0)

def polyline_example():
    """Returns a continuous tube connecting multiple points."""
    points = [
        (-1, 0, 0), (0, 1, 0), (1, 0, 0), (2, 1, 0), (2, -1, 0)
    ]
    return polyline(points, radius=0.1)

def polycurve_example():
    """Returns a smooth loop passing near control points."""
    points = [
        (0, 0, 0), (1, 2, 0), (2, 0, 0), (3, 2, 0), (4, 0, 0)
    ]
    # Create a ribbon by extruding the curve
    return polycurve(points, radius=0.1, closed=False)

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Primitive Examples ---")

    examples = {
        "sphere": sphere_example,
        "box": box_example,
        "cylinder": cylinder_example,
        "torus": torus_example,
        "cone": cone_example,
        "hex": hex_prism_example,
        "pyramid": pyramid_example,
        "curve": curve_example,
        "circle": circle_example,
        "rectangle": rectangle_example,
        "triangle": triangle_example,
        "trapezoid": trapezoid_example,
        "polyline": polyline_example,
        "polycurve": polycurve_example,
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