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
    Renders a default example when the script is run directly.
    Change the `example_name` here to view different examples.
    """
    print("--- SDForge Operation Examples ---")
    
    examples = {
        "union": union_example,
        "intersection": intersection_example,
        "difference": difference_example,
        "smooth_union": smooth_union_example,
    }
    
    # --- Select the example to render ---
    example_name = "smooth_union"  # <-- CHANGE THIS VALUE
    # ------------------------------------

    print(f"Rendering: {example_name.replace('_', ' ').title()} Example")
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