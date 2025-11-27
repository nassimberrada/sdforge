import sys
from sdforge import box, sphere, circle, rectangle, X

def round_example():
    """Rounds the sharp edges of a box."""
    b = box(size=(1.5, 1.0, 0.5))
    return b.round(0.2)

def masked_round_example():
    """
    Demonstrates rounding only specific parts of an object.
    Here, only one corner of the box is rounded.
    """
    b = box(size=1.5)
    
    # Mask selects only the corner area at (+X, +Y, +Z)
    # A sphere placed at that corner serves as the selection volume.
    mask = sphere(0.8).translate((0.75, 0.75, 0.75))
    
    # Radius is 0.3 inside the mask, 0.0 outside.
    return b.round(0.3, mask=mask, mask_falloff=0.1)

def shell_example():
    """Creates a hollow shell from a sphere."""
    s = sphere(radius=1.0)
    # The parameter controls the thickness of the shell.
    return s.shell(0.1)

def masked_shell_example():
    """
    Demonstrates a shell with variable thickness.
    The object transitions from a thick wall to a thin (or zero) wall.
    """
    s = sphere(1.0)
    
    # Mask covers the top half.
    # Top half: Thickness 0.1
    # Bottom half: Thickness 0.0 (infinitely thin surface)
    mask = box(2.0).translate((0, 1.0, 0))
    
    # Note: A thickness of 0.0 means the surface exists but has no volume.
    # In the viewer, you will see the wall taper until it is paper-thin.
    return s.shell(0.1, mask=mask, mask_falloff=0.2)

def extrude_example():
    """Extrudes a 2D circle into a 3D cylinder."""
    c = circle(radius=0.8)
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
        "masked_round": masked_round_example,
        "shell": shell_example,
        "masked_shell": masked_shell_example,
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