import sys
from sdforge import sphere, box, cylinder, torus, cone

def sphere_example():
    """Returns a simple sphere scene."""
    s = sphere(r=1.0)
    return s

def box_example():
    """Returns a simple box scene."""
    return box(size=(1.5, 1.0, 0.5), radius=0.1)

def cylinder_example():
    """Returns a simple cylinder scene."""
    return cylinder(radius=0.5, height=1.5)

def torus_example():
    """Returns a simple torus scene."""
    return torus(major=1.0, minor=0.25)
    
def cone_example():
    """Returns a frustum (capped cone) scene."""
    return cone(height=1.2, radius1=0.6, radius2=0.2)

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