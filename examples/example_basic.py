from sdforge import *

def main():
    """
    This function defines the SDF model.
    For hot-reloading to work, it must return the final SDF object.
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