from sdforge import *

def main():
    """
    Demonstrates creating 3D shapes from 2D profiles.

    This example shows how to:
    - Create 2D primitives like `circle` and `rectangle`.
    - Combine them using standard boolean operators.
    - Use `.extrude()` to give the 2D profile depth, creating a 3D object.
    - Use `.revolve()` to spin the 2D profile around the Y-axis, creating a
      lathe-like object.
    """

    # --- 1. Define a 2D Profile ---
    # Create a 2D profile using 2D primitives and operators.
    # This shape exists on the XY plane.
    profile = rectangle((1.5, 2.0)) - circle(0.8)
    profile = profile.round(0.1)
    
    # --- 2. Extrude the Profile ---
    # `.extrude()` gives the 2D shape a thickness along the Z-axis.
    extruded_shape = profile.extrude(0.5)


    # --- 3. Revolve the Profile ---
    # `.revolve()` spins the 2D shape around the Y-axis.
    # First, we move the profile away from the rotation axis (the Y-axis).
    revolved_shape = profile.translate(X * 2.0).revolve()
    

    # Combine the two generated 3D shapes into a single scene
    # and move them to be side-by-side.
    final_scene = extruded_shape.translate(-X * 2.0) | revolved_shape

    return final_scene

if __name__ == '__main__':
    model = main()
    if model:
        model.render(watch=True)