from sdforge import *

def main():
    """
    Demonstrates static and animated camera controls.

    This example shows how to:
    - Return a tuple `(sdf_object, camera_object)` from the main function.
    - Define a static camera with a fixed position and target.
    - Define an animated camera whose properties are GLSL expressions
      using the `u_time` uniform.
    """

    # A simple shape
    shape = sphere(1) & box(1.5)

    # --- Camera ---
    # You can define a camera and animate its properties using GLSL expressions.
    # The `u_time` uniform is available for animations.
    
    # 1. Static Camera (uncomment to use)
    # cam = Camera(position=(5, 4, 5), target=(0, 0, 0))

    # 2. Animated Orbiting Camera
    cam = Camera(
        position=(
            "5.0 * sin(u_time * 0.5)",
            "3.0",
            "5.0 * cos(u_time * 0.5)"
        ),
        target=(0, 0, 0)
    )

    # When hot-reloading, changes to both the shape and the camera will be applied.
    return shape, cam

if __name__ == "__main__":
    # The render function will automatically use the camera if it's returned by main.
    result = main()
    if result:
        sdf_object, camera_object = result
        sdf_object.render(camera=camera_object, watch=True)