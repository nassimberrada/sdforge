import sys
from sdforge import (
    sphere, box, cylinder, torus, cone, hex_prism, pyramid, 
    curve, circle, rectangle, triangle, trapezoid, polyline, polycurve,
    Forge, Sketch, X, Y, Z
)

def sphere_example(): return sphere(radius=1.0)
def box_example(): return box(size=(1.5, 1.0, 0.5)).round(0.1)
def cylinder_example(): return cylinder(radius=0.5, height=1.5)
def torus_example(): return torus(radius_major=1.0, radius_minor=0.25)
def cone_example(): return cone(height=1.2, radius_base=0.6, radius_top=0.2)
def hex_prism_example(): return hex_prism(radius=1.0, height=0.5)
def pyramid_example(): return pyramid(height=1.2)
def curve_example(): return curve(p0=(-1.0, 0.0, 0.0), p1=(0.0, 1.5, 0.0), p2=(1.0, 0.0, 0.0), radius=0.1)
def circle_example(): return circle(radius=1.0)
def rectangle_example(): return rectangle(size=(1.5, 1.0))
def triangle_example(): return triangle(radius=1.0)
def trapezoid_example(): return trapezoid(bottom_width=1.5, top_width=0.8, height=1.0)

def polyline_example():
    points = [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (2, 1, 0), (2, -1, 0)]
    return polyline(points, radius=0.1)

def polycurve_example():
    points = [(0, 0, 0), (1, 2, 0), (2, 0, 0), (3, 2, 0), (4, 0, 0)]
    return polycurve(points, radius=0.1, closed=False)

def simple_forge_example(): return Forge("length(p) - 1.0")
def uniform_forge_example():
    glsl_code = """
        vec3 q = abs(p) - u_size;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
    """
    return Forge(glsl_code, uniforms={'u_size': 0.8})

def simple_profile_example():
    return (Sketch(start=(0, 0))
         .line_to(1.0, 0).line_to(1.0, 0.2).line_to(0.2, 0.2).line_to(0.2, 1.0)
         .curve_to(0, 1.2, control=(0.0, 1.0)).close()
         .to_sdf(stroke_radius=0.05))

def extruded_sketch_example():
    profile = (Sketch(start=(0, 0))
         .line_to(1.0, 0).line_to(1.0, 0.2).line_to(0.2, 0.2).line_to(0.2, 1.0)
         .curve_to(0, 1.2, control=(0.0, 1.0)).close()
         .to_sdf(stroke_radius=0.05))
    return profile.extrude(height=0.5)

def revolved_sketch_example():
    vase_profile = (Sketch(start=(0.5, 0))
        .line_to(1.0, 0.2).curve_to(0.6, 1.0, control=(1.5, 0.5))
        .line_to(0.8, 1.5).to_sdf(stroke_radius=0.02))
    return vase_profile.revolve()

def main():
    print("--- SDForge Geometry Examples ---")
    examples = {
        "sphere": sphere_example, "box": box_example, "cylinder": cylinder_example,
        "torus": torus_example, "cone": cone_example, "hex": hex_prism_example,
        "pyramid": pyramid_example, "curve": curve_example,
        "circle": circle_example, "rectangle": rectangle_example, "triangle": triangle_example,
        "trapezoid": trapezoid_example, "polyline": polyline_example, "polycurve": polycurve_example,
        "forge_simple": simple_forge_example, "forge_uniform": uniform_forge_example,
        "sketch_profile": simple_profile_example, "sketch_extrude": extruded_sketch_example,
        "sketch_revolve": revolved_sketch_example,
    }

    if len(sys.argv) < 2:
        print("Available examples:", ", ".join(examples.keys()))
        return

    func = examples.get(sys.argv[1])
    if func: 
        scene = func()
        scene.render()
    else: print(f"Example '{sys.argv[1]}' not found.")

if __name__ == "__main__":
    main()