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

## Hello Forge

Define a simple shape and open a real-time preview window with a couple lines:

```python
from sdforge import *

# A sphere intersected with a box
shape = sphere(1) & box(1.5)

# Render a preview in a native window.
shape.render()
```

## Basics

### Defining & Combining Shapes
You can create complex objects by starting with primitives and combining them with Python's bitwise operators: `|` for union, `&` for intersection, and `-` for difference.

```python
from sdforge import *

# A sphere intersected with a box
f = sphere(1) & box(1.5)

# Subtract three cylinders along each axis
c = cylinder(0.5)
f -= c.orient(X) | c.orient(Y) | c.orient(Z)

f.render()
```

### Live Preview & Hot-Reloading
For an interactive workflow, wrap your scene definition in a function and use the `watch` flag. When you save any changes to the file, the preview window will automatically update.

```python
from sdforge import *

def main():
    f = sphere(1) & box(1.5)
    return f

if __name__ == '__main__':
    f = main()

    # The view will update every time you save the source file.
    f.render(watch=True)
```

### Saving to a Mesh File
You can save any **static (non-animated)** model to an `.stl` file for 3D printing or for use in other software. The `.save()` method uses the Marching Cubes algorithm to generate a mesh from the SDF.

```python
# Save the model to an .stl file.
f.save('model.stl', samples=2**24) # Higher samples = more detail
```

## Advanced

### GLSL Injection`
Many functions and object properties can accept GLSL expressions as strings. For e.g you can use the `u_time` uniform, which contains the elapsed time in seconds, to create animations.

```python
from sdforge import *

# Animate a box's size using the u_time uniform
f = box(size="0.5 + 0.3 * sin(u_time)")

f.render()
```

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

f.render()
```

### Custom Camera Controls
You can override the default mouse-orbit camera to create cinematic animations or set a static viewpoint. The `Camera` object accepts GLSL expressions for its position and target, which will be updated every frame.

```python
from sdforge import *

def main():
    shape = sphere(1) & box(1.5)

    # An animated camera that orbits around the origin
    cam = Camera(
        position=(
            "5.0 * sin(u_time * 0.5)",
            "3.0",
            "5.0 * cos(u_time * 0.5)"
        ),
        target=(0, 0, 0)
    )
    return shape, cam

if __name__ == '__main__':
    sdf_object, camera_object = main()
    sdf_object.render(camera=camera_object, watch=True)
```

### Custom Light Controls
You can customize the scene's light, including light position, ambient light, and shadow softness. The `Light` object also accepts GLSL expressions for its properties.

```python
from sdforge import *

def main():
    shape = sphere(1) - cylinder(0.5)
    camera = Camera(
        position=(
            "5.0 * sin(u_time * 0.5)", 
            "3.0", 
            "5.0 * cos(u_time * 0.5)"
        )
    )

    # An animated light source with soft shadows
    light = Light(
        position=(
            "8.0 * sin(u_time * 0.3)",
            "5.0",
            "8.0 * cos(u_time * 0.3)"
        ),
        ambient_strength=0.05,
        shadow_softness="8.0 + 7.0 * sin(u_time * 0.7)"
    )

    return shape, camera, light

if __name__ == '__main__':
    sdf_obj, cam_obj, light_obj = main()
    sdf_obj.render(camera=cam_obj, light=light_obj, watch=True)
```

### Assigning Materials
You can assign a unique color to any object or group of objects using the `.color()` method. The renderer will automatically handle combining the shapes and their materials correctly.

```python
from sdforge import *

red_sphere = sphere(0.8).color(1, 0, 0)
blue_box = box(1.2).color(0, 0, 1)

# The union operation will correctly preserve the material of the closest surface
model = red_sphere | blue_box.translate(X * 0.5)

# You can also set a custom background color
model.render(bg_color=(0.1, 0.2, 0.3))
```

### Recording to Video
You can record an interactive session to an MP4 video file by passing the `record` argument to the `.render()` method. This requires the optional `[record]` dependencies.

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

To enable optional video recording features, install with the `[record]` extra:

```bash
pip install sdforge[record]
```

## Acknowledgements

This project is inspired by the simplicity and elegant API of Michael Fogleman's [fogleman/sdf](https://github.com/fogleman/sdf) library. SDF Forge aims to build on that foundation by adding a real-time, interactive GLSL-powered renderer.