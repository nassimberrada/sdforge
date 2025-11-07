from sdforge import *

def main():
    """
    Demonstrates injecting animated GLSL expressions into shape parameters.

    This example shows how to:
    - Pass a string containing a GLSL expression (e.g., using `u_time`)
      as a parameter to a primitive like `box`.
    - Animate properties of shapes over time.
    - Note that models with animated parameters cannot be saved to a mesh file.
    """
    # Create a box whose size is a GLSL expression string.
    # This animates the size smoothly between 0.1 and 0.9.
    animated_box = box(size="0.5 + 0.3 * sin(u_time * 0.6)")

    # The rest of the API works exactly the same.
    f = sphere(1.4) | animated_box.translate(offset=(1, 1, 1.5))
    
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    
    return f

if __name__ == "__main__":
    sdf_object = main()
    if sdf_object:
        sdf_object.render(watch=True)

        # NOTE: If you uncomment the line below, it will raise a TypeError
        # because the 'animated_box' has a dynamic size.
        # sdf_object.save('animated_model.stl')