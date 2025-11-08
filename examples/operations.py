import sys
from sdforge import sphere, box

def union_example():
    """A sphere and a box joined together."""
    s = sphere(r=0.8)
    b = box(size=(1.5, 0.5, 0.5))
    return s | b

def intersection_example():
    """A lens shape created by intersecting two spheres."""
    s1 = sphere(r=1.0).translate((-0.5, 0, 0))
    s2 = sphere(r=1.0).translate((0.5, 0, 0))
    return s1 & s2

def difference_example():
    """A box with a sphere carved out of it."""
    b = box(size=1.5)
    s = sphere(r=1.0)
    return b - s

def smooth_union_example():
    """Two spheres smoothly blended together."""
    s1 = sphere(r=0.7).translate((-0.5, 0, 0))
    s2 = sphere(r=0.7).translate((0.5, 0, 0))
    # The 'k' parameter controls the smoothness of the blend.
    return s1.union(s2, k=0.3)

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