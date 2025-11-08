import sys
from sdforge import box, Param, Y
import numpy as np

def main():
    """
    Demonstrates using interactive parameters.
    
    While this script is running, a UI would typically appear allowing you
    to drag sliders to change the model in real-time. This example sets up
    the backend for that functionality.
    """
    print("--- SDForge Interactive Param Example ---")

    # Create Param objects to control different aspects of the scene.
    # Param(name, default_value, min_value, max_value)
    p_size = Param("Box Size", 0.8, 0.2, 2.0)
    p_radius = Param("Corner Radius", 0.1, 0.0, 0.5)
    p_twist = Param("Twist", 0.0, -10.0, 10.0)

    # Use the Param objects just like regular numbers.
    scene = box(size=p_size, radius=p_radius).twist(p_twist)
    
    # You can also use them in GLSL expressions for animation.
    # Note: `save()` will fail if string expressions are used.
    # animated_box = box(1.0).translate((np.sin(time) * p_size, 0, 0))

    # The renderer will automatically create uniforms for each Param.
    return scene

if __name__ == "__main__":
    scene = main()
    # To see the UI sliders, an ImGui-based renderer would be needed.
    # This default renderer will use the default Param values.
    scene.render()