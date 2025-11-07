from sdforge import *

def main():
    """
    Demonstrates advanced shaping operators like elongate, displace, and extrude.
    """

    # --- Elongate and XOR ---
    # Start with a base shape, a torus
    # Elongate it along the X and Z axes to make a 'frame' shape
    shape = torus(major=1.0, minor=0.2).elongate((0.8, 0, 0.8))

    # Use the XOR operator with a box to create an interesting intersection
    shape = shape.xor(box(1.5))
    
    # --- Displacement ---
    # Apply a procedural displacement using a GLSL expression.
    # The 'p' variable (vec3) is the point in space being sampled.
    displacement = "sin(p.x * 20.0) * sin(p.y * 20.0) * sin(p.z * 20.0) * 0.05"
    displaced_shape = shape.displace(displacement)
    
    # --- Revolution (new) ---
    # To revolve, we first need a 2D SDF. We can define one using `Forge`.
    # This 2D shape defines a profile that will be spun around the Y axis.
    profile_2d = Forge("""
        vec2 q = p.xy - vec2(0.8, 0); // Move the center of the shape off the rotation axis
        float a = atan(q.y, q.x);
        float r = length(q);
        return r - (0.3 + 0.1*sin(a*5.0)); // A 2D flower shape
    """)
    
    # Revolve the 2D profile to create a 3D object
    revolved_shape = profile_2d.revolve() + (-X * 4)

    # --- Limited Repetition ---
    # Create a base object and repeat it a limited number of times.
    base_obj = box((0.5, 2.0, 0.5), radius=0.1)
    repeated_obj = base_obj.limited_repeat(spacing=(0.8, 0, 0), limits=(3, 0, 0))
    repeated_obj += (X * 4)

    # Combine all the objects into one scene
    final_scene = displaced_shape | revolved_shape | repeated_obj

    return final_scene

# This is the standard way to run the script.
if __name__ == '__main__':
    sdf_object = main()
    if sdf_object:
        # The render call is now a simple, blocking function call.
        # It will run until you close the window.
        sdf_object.render(watch=True)

        # NOTE: Saving is not possible for shapes using GLSL expressions
        # via `Forge` or `displace`.
        # repeated_obj.save('repeated_obj.stl') # This would work!