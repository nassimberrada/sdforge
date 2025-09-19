<p align="center">
  <picture>
    <source srcset="./assets/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="./assets/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="./assets/logo_light.png" alt="SDForge Logo" height="200">
  </picture>
</p>

SDF Forge is a Python library for creating 3D models using Signed Distance Functions (SDFs). It provides a real-time, interactive rendering experience in a native desktop window, powered by GLSL raymarching.

## Features

- **Simple, Pythonic API:** Define complex shapes by combining primitives using standard operators (`|`, `-`, `&`).
- **Real-time Rendering with Hot-Reloading:** Get instant visual feedback in a lightweight native window powered by `moderngl` and `glfw`.
- **Mesh Exporting:** Save your creations as `.stl` files for 3D printing or use in other software.
- **Flexible Scene Construction:** Write custom SDF logic directly in GLSL, easily assign different materials to individual objects, etc.

## Quick Start

```python
from sdforge import *

# A sphere intersected with a box
f = sphere(1) & box(1.5)

# Subtract three cylinders along each axis
c = cylinder(0.5)
f -= c.orient(X) | c.orient(Y) | c.orient(Z)

# Render a live preview in a native window.
# With watch=True, the view will update when you save the file.
f.render(watch=True)
```

## Advanced

### Custom GLSL with `Forge`

For complex or highly-performant shapes, you can write GLSL code directly. This object integrates perfectly with the rest of the API.

```python
from sdforge import *

# A standard library primitive
s = sphere(1.2)

# A custom shape defined with GLSL
# 'p' is the vec3 point in space
custom_twist = Forge("""
    float k = 10.0;
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xz,p.y);
    return length(q) - 0.5;
""")

f = s - custom_twist

# Rendering and saving works out of the box
f.render()
f.save('example_forge.stl')
```

### Camera Controls

You can override the default mouse-orbit camera to create cinematic animations or set a static viewpoint. The `Camera` object accepts GLSL expressions for its position and target, which will be updated every frame.

To use a custom camera, have your `main` function return a tuple containing your `SDFObject` and your `Camera` object.

```python
from sdforge import *

def main():
    # A simple shape to look at
    shape = sphere(1) & box(1.5)

    # An animated camera that orbits around the origin
    cam = Camera(
        position=(
            "5.0 * sin(u_time * 0.5)",
            "3.0",
            "5.0 * cos(u_time * 0.5)"
        ),
        target=(0, 0, 0) # Look at the center
    )

    # For hot-reloading to work, the main function must return the shape and camera
    return shape, cam

if __name__ == '__main__':
    sdf_object, camera_object = main()
    sdf_object.render(camera=camera_object, watch=True)
```

### Light Controls

You can customize the scene's lighting, including light position, ambient light, and shadow softness. The `Light` object accepts GLSL expressions for its properties, which will be updated every frame.

To use custom lighting, your `main` function can return a tuple containing your `SDFObject`, your `Camera` object, and your `Light` object.

```python
from sdforge import *

# A simple shape to look at
shape = sphere(1) - cylinder(0.5)

# A standard orbiting camera
cam = Camera(position=("5.0 * sin(u_time * 0.5)", "3.0", "5.0 * cos(u_time * 0.5)"))

# An animated light source with soft shadows
lighting = Light(
    position=(
        "8.0 * sin(u_time * 0.3)",
        "5.0",
        "8.0 * cos(u_time * 0.3)"
    ),
    ambient_strength=0.05,
    shadow_softness="8.0 + 7.0 * sin(u_time * 0.7)"
)

# For hot-reloading, return all scene objects from main
def main():
    return shape, cam, lighting

if __name__ == '__main__':
    sdf_obj, cam_obj, light_obj = main()
    sdf_obj.render(camera=cam_obj, lighting=light_obj, watch=True)
```

### Material Assignment

You can assign a unique color to any object or group of objects using the `.color()` method. The renderer will automatically handle combining the shapes and their materials correctly.

```python
from sdforge import *

# Define shapes with different colors
red_sphere = sphere(0.8).color(1, 0, 0)
blue_box = box(1.2).color(0, 0, 1)

# Combine colored objects
# The union operation will correctly preserve the material of the closest surface
model = red_sphere | blue_box.translate(X * 0.5)

# You can also set a custom background color
model.render(bg_color=(0.1, 0.2, 0.3))
```

### Render to File

You can save any static (non-animated) SDF model to an `.stl` file for 3D printing or use in other software. The `.save()` method uses the Marching Cubes algorithm to generate a mesh from the SDF.

```python
from sdforge import *

# A sphere intersected with a box
f = sphere(1) & box(1.5)

# Subtract three cylinders along each axis
c = cylinder(0.5)
f -= c.orient(X) | c.orient(Y) | c.orient(Z)

# Save the model to a file
f.save('model.stl', samples=2**24) # Higher samples = more detail
```

### Record Render

You can record the interactive session to an MP4 video file by passing the `record` argument to the `.render()` method. This requires the optional `[record]` dependencies.

```python
from sdforge import *

# Animate a box size using the u_time uniform
f = box(size="0.5 + 0.3 * sin(u_time)")

# Render and record the output to a video file.
# Close the window to stop the recording.
f.render(record="animated_box.mp4")
```

## Installation

The library and its core dependencies can be installed using pip:

```bash
pip install sdforge
```

To enable optional video recording features, install the `[record]` extra:

```bash
pip install sdforge[record]
```

## Acknowledgements

This project is inspired by the simplicity and elegant API of Michael Fogleman's [fogleman/sdf](https://github.com/fogleman/sdf) library. SDF Forge aims to build on that foundation by adding a real-time, interactive GLSL-powered renderer.