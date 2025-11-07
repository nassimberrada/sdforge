from sdforge import *

def main():
    """
    Demonstrates the basic concepts of SDF modeling.

    This example shows how to:
    - Create primitive shapes like `sphere` and `box`.
    - Combine shapes using boolean operators: intersection (`&`),
      union (`|`), and difference (`-`).
    - Apply transformations like `orient`.
    """
    # A sphere intersected with a box
    f = sphere(1) & box(1.5)

    # Subtract three cylinders along each axis
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)

    return f

# This is now the clean, standard way to run the script.
if __name__ == "__main__":
    sdf_object = main()
    if sdf_object:
        # The render call is now a simple, blocking function call.
        # It will run the server until you press Ctrl+C in the terminal.
        sdf_object.render(watch=True)