import sys
from sdforge import box, sphere, cylinder, X

def simple_material_example():
    """
    Demonstrates applying different colors to objects.
    """
    # A blue sphere is subtracted from a red box.
    red_box = box(1.5).round(0.1).color(1.0, 0.2, 0.2)
    blue_sphere = sphere(radius=1.2).color(0.3, 0.5, 1.0)
    
    scene = red_box - blue_sphere
    return scene

def masked_material_example():
    """
    Demonstrates applying a color to a specific region using a mask.
    This creates a 'decal' effect or a pattern.
    """
    # Base object: White box
    base = box(2.0).color(0.9, 0.9, 0.9)
    
    # Mask: A cylinder running through the box along X
    # We want to paint the box red where this cylinder intersects it.
    mask_shape = cylinder(radius=0.5, height=3.0).rotate(X, 1.57).translate((0, 0.5, 0))
    
    # Apply red color using the mask.
    # Where mask < 0 (inside cylinder), color is Red.
    # Elsewhere, it keeps the previous color (White).
    scene = base.color(1.0, 0.1, 0.1, mask=mask_shape)
    
    return scene

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Material Examples ---")
    
    examples = {
        "simple": simple_material_example,
        "masked": masked_material_example,
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