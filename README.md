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
- **Real-time Native Rendering:** Get instant visual feedback in a lightweight native window powered by `moderngl` and `glfw`.
- **Hot-Reloading:** Save your Python script and the 3D view updates instantly, without restarting.
- **Mesh Exporting:** Save your creations as `.stl` files for 3D printing or use in other software.
- **Custom GLSL with `Forge`:** Write custom SDF logic directly in GLSL and integrate it into the Python workflow.

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