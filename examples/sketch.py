import sys
from sdforge import Sketch, X

def simple_profile_example():
    """
    Creates a 2D profile using the Sketch API.
    It looks like a simple bracket.
    """
    s = (Sketch(start=(0, 0))
         .line_to(1.0, 0)         # Bottom edge
         .line_to(1.0, 0.2)       # Up
         .line_to(0.2, 0.2)       # Inner bottom
         .line_to(0.2, 1.0)       # Inner vertical
         .curve_to(0, 1.2, control=(0.0, 1.0)) # Rounded top corner
         .close()                 # Back to (0,0) with a straight line
         .to_sdf(stroke_radius=0.05)
    )
    return s

def extruded_sketch_example():
    """
    Creates the same profile but extrudes it into a 3D object.
    """
    profile = (Sketch(start=(0, 0))
         .line_to(1.0, 0)
         .line_to(1.0, 0.2)
         .line_to(0.2, 0.2)
         .line_to(0.2, 1.0)
         .curve_to(0, 1.2, control=(0.0, 1.0))
         .close()
         .to_sdf(stroke_radius=0.05)
    )

    # Extrude creates a 3D shape with height 0.5
    return profile.extrude(height=0.5)

def revolved_sketch_example():
    """
    Creates a vase-like profile and revolves it.
    Note: Revolve happens around the Y axis, so the profile
    should be defined in XY with X > 0.
    """
    # Start slightly offset from X=0 to make a hollow vase
    vase_profile = (Sketch(start=(0.5, 0))
        .line_to(1.0, 0.2)        # Base flare
        .curve_to(0.6, 1.0, control=(1.5, 0.5)) # Body curve
        .line_to(0.8, 1.5)        # Neck
        .to_sdf(stroke_radius=0.02)
    )

    return vase_profile.revolve()

def curved_close_example():
    """
    Demonstrates using a curved closure for a sketch.
    Creates a rounded triangle-like shape.
    """
    s = (Sketch(start=(0, 0))
         .line_to(2.0, 0)           # Bottom edge
         .line_to(1.0, 1.5)         # Angled edge to top
         .close(curve_control=(0.0, 0.75)) # Curve back to (0,0)
         .to_sdf(stroke_radius=0.05)
    )
    return s

def main():
    """
    Renders a sketch example based on a command-line argument.
    """
    print("--- SDForge Sketch API Examples ---")

    examples = {
        "profile": simple_profile_example,
        "extrude": extruded_sketch_example,
        "revolve": revolved_sketch_example,
        "curved_close": curved_close_example, # Added new example
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