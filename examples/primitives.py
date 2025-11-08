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
    Renders a default example when the script is run directly.
    Change the `example_name` here to view different examples.
    """
    print("--- SDForge Primitive Examples ---")
    
    examples = {
        "sphere": sphere_example,
        "box": box_example,
        "cylinder": cylinder_example,
        "torus": torus_example,
        "cone": cone_example,
    }
    
    # --- Select the example to render ---
    example_name = "box"  # <-- CHANGE THIS VALUE
    # ------------------------------------
    
    print(f"Rendering: {example_name.title()} Example")
    scene_func = examples.get(example_name)
    
    if scene_func:
        scene = scene_func()
    else:
        scene = None
        print(f"Example '{example_name}' not found.")

    if scene:
        scene.render()
    else:
        print("The selected example is not yet implemented or not found.")


if __name__ == "__main__":
    main()