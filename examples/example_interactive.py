from sdforge import *

def main():
    """
    Demonstrates creating real-time interactive models using UI parameters.

    This example shows how to:
    - Use the `Param` object to create a named parameter with a default
      value and a min/max range.
    - Use the `Param` object as a value for primitive or operator arguments.
    - The renderer automatically creates UI sliders for each `Param`, allowing
      real-time updates to the model in the viewer.
    """

    # Create interactive parameters for the box
    size_x = Param("Size X", 1.5, 0.1, 3.0)
    size_y = Param("Size Y", 1.5, 0.1, 3.0)
    size_z = Param("Size Z", 1.5, 0.1, 3.0)
    
    # And for the rounding radius
    rounding = Param("Rounding", 0.2, 0.0, 1.0)

    # Use the parameters to define the shape
    shape = box(x=size_x, y=size_y, z=size_z, radius=rounding)

    # Create a parameter for the noise displacement
    noise_strength = Param("Noise Strength", 0.0, 0.0, 0.5)
    noise_scale = Param("Noise Scale", 10.0, 1.0, 50.0)

    # Apply noise displacement using the parameters
    shape = shape.displace_by_noise(scale=noise_scale, strength=noise_strength)

    return shape

if __name__ == "__main__":
    model = main()
    if model:
        # The renderer will automatically detect the Param objects
        # and create UI sliders for them.
        model.render(watch=True)