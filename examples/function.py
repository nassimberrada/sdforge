import sys
import sdforge as sdf

def sphere_example():
    """
    A simple algebraic sphere. 
    Equation: x^2 + y^2 + z^2 - r^2 = 0
    """
    radius = 2.0
    math_expr = f"p.x*p.x + p.y*p.y + p.z*p.z - {radius * radius}"
    
    sphere = sdf.Function(math_expr, safety=0.5)
    
    shape = sphere.intersection(sdf.box(5.0)).color((0.2, 0.6, 0.9))
    return shape, sdf.Camera(zoom=1.2)


def hyperboloid_example():
    """
    Hyperboloid of One Sheet.
    A classic ruled surface often used in cooling towers and architecture.
    Equation: x^2 + y^2 - z^2 - 1 = 0
    """
    math_expr = "p.x*p.x + p.y*p.y - p.z*p.z - 1.0"
    hyperboloid = sdf.Function(math_expr, safety=0.5)
    
    shape = hyperboloid.intersection(sdf.box((6, 6, 6)))
    shape = shape.rotate(sdf.X, 0.6).color((0.9, 0.4, 0.1))
    
    return shape, sdf.Camera(zoom=0.8)


def gyroid_example():
    """
    The Gyroid: A triply periodic minimal surface.
    Equation: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
    """
    math_expr = "sin(p.x)*cos(p.y) + sin(p.y)*cos(p.z) + sin(p.z)*cos(p.x)"
    
    gyroid = sdf.Function(math_expr, safety=0.25)
    
    shape = gyroid.intersection(sdf.sphere(5.0)).color((0.2, 0.8, 0.4))
    
    return shape, sdf.Camera(zoom=0.6)


def torus_knot_example():
    """
    An algebraic approximation of a Torus Knot.
    This demonstrates how complex you can make the inline math!
    """
    math_expr = """
        float r1 = 2.0;  // Major radius
        float r2 = 0.5;  // Minor radius
        float freq = 3.0; // Number of twists
        
        // Convert Cartesian (x,y,z) to Cylindrical (r, theta, y)
        float r = length(p.xz);
        float theta = atan(p.z, p.x);
        
        // Calculate the distance to the twisted core
        float core_x = r - r1;
        float core_y = p.y;
        
        // Apply the twist
        float current_angle = theta * freq;
        float final_x = core_x * cos(current_angle) - core_y * sin(current_angle);
        float final_y = core_x * sin(current_angle) + core_y * cos(current_angle);
        
        // Distance to the tube surface
        float d = length(vec2(final_x, final_y)) - r2;
        return d;
    """
    
    knot = sdf.Function(math_expr, safety=0.4)
    shape = knot.intersection(sdf.box(6.0)).color((0.8, 0.2, 0.8))
    
    return shape, sdf.Camera(zoom=0.8)


def main():
    print("--- SDForge Function Surface Examples ---")
    
    examples = {
        "sphere": sphere_example,
        "hyperboloid": hyperboloid_example,
        "gyroid": gyroid_example,
        "knot": torus_knot_example
    }
    
    if len(sys.argv) < 2:
        print("\nPlease provide the name of an example to run.")
        print("Available examples:")
        for key in examples:
            print(f"  - {key}")
        print(f"\nUsage: python {sys.argv[0]} <example_name>")
        return

    example_name = sys.argv[1].lower()
    scene_func = examples.get(example_name)
    
    if not scene_func:
        print(f"\nError: Example '{example_name}' not found.")
        print("Available examples are:")
        for key in examples:
            print(f"  - {key}")
        return

    print(f"Rendering: {example_name.title()} Example")
    
    # Run the selected example
    result = scene_func()
    
    # Render the result
    if isinstance(result, tuple):
        scene, cam = result
        scene.render(camera=cam)
    else:
        result.render()

if __name__ == "__main__":
    main()