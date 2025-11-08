from sdforge import sphere

def sphere_example():
    """Returns a simple sphere scene."""
    s = sphere(r=1.0)
    return s

def main():
    """
    Renders a default example when the script is run directly.
    Change the function call here to view different examples.
    """
    print("--- SDForge Primitive Examples ---")
    
    # --- Select the example to render ---
    print("Rendering: Sphere Example")
    scene = sphere_example()
    # ------------------------------------

    if scene:
        scene.render()
    else:
        print("The selected example is not yet implemented.")


if __name__ == "__main__":
    main()