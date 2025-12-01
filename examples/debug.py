import sys
from sdforge import box, sphere, Debug

def normals_debug_example():
    """Visualizes the surface normals as colors."""
    scene = box(1.5).round(0.1) - sphere(radius=1.2)
    debug = Debug('normals')
    return scene, debug

def steps_debug_example():
    """Visualizes the number of raymarching steps."""
    scene = box(1.5).round(0.1) - sphere(radius=1.2)
    debug = Debug('steps')
    return scene, debug

def slice_debug_example():
    """
    Visualizes a 2D cross-section of the distance field.

    Orange = Positive distance (Outside)
    Blue   = Negative distance (Inside)
    White  = Surface boundary
    """
    # A hollow sphere cut in half
    scene = sphere(1.0).shell(0.1) - box(1.0).translate((0.5, 0, 0))

    # Slice through the XY plane at Z=0, with a view scale of 3.0 units
    debug = Debug('slice', plane='xy', slice_height=0.0, view_scale=3.0)
    return scene, debug

def main():
    """Renders a debug example based on a command-line argument."""
    print("--- SDForge Debug Examples ---")

    examples = {
        "normals": normals_debug_example,
        "steps": steps_debug_example,
        "slice": slice_debug_example,
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

    scene, debug = None, None
    for item in result:
        from sdforge.api.core import SDFNode
        if isinstance(item, SDFNode): scene = item
        if isinstance(item, Debug): debug = item

    if scene:
        scene.render(debug=debug)
    else:
        print("Error: Example function did not return a scene object.")


if __name__ == "__main__":
    main()