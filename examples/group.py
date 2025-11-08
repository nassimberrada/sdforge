import sys
import numpy as np
from sdforge import box, sphere, Group, Y

def group_transform_example():
    """
    Demonstrates applying a single transformation to a group of objects.
    """
    # Create two separate objects, offset from the origin.
    b = box(size=(1.5, 0.5, 0.5), radius=0.1).translate((-1, 0, 0))
    s = sphere(0.5).translate((1, 0, 0))
    
    # Combine them into a group.
    g = Group(b, s)
    
    # Now, a single rotation is applied to both objects simultaneously,
    # rotating them as a single unit around the world origin.
    scene = g.rotate(Y, np.pi / 4)
    
    return scene

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Grouping Examples ---")
    
    examples = {
        "transform": group_transform_example,
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
    result = scene_func()
    
    if isinstance(result, tuple):
        scene, cam = result
        scene.render(camera=cam)
    else:
        scene = result
        scene.render()

if __name__ == "__main__":
    main()