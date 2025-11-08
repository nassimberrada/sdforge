import sys
from sdforge import box, sphere, circle, rectangle, X

def round_example():
    """Rounds the sharp edges of a box."""
    b = box(size=(1.5, 1.0, 0.5))
    return b.round(0.2)

def shell_example():
    """Creates a hollow shell from a sphere."""
    s = sphere(r=1.0)
    # The parameter controls the thickness of the shell.
    return s.shell(0.1)

def extrude_example():
    """Extrudes a 2D circle into a 3D cylinder."""
    c = circle(r=0.8)
    return c.extrude(1.5)

def revolve_example():
    """Revolves a 2D profile into a 3D vase-like shape."""
    # Create a 2D profile by combining rectangles.
    # It must be offset from the Y-axis (the axis of revolution).
    r1 = rectangle(size=(0.4, 1.0)).translate((0.7, 0, 0))
    r2 = rectangle(size=(0.8, 0.2)).translate((0.5, 0, 0))
    profile = r1 | r2
    return profile.revolve()

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Shaping Examples ---")
    
    examples = {
        "round": round_example,
        "shell": shell_example,
        "extrude": extrude_example,
        "revolve": revolve_example,
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