<p align="center">
  <picture>
    <source srcset="./assets/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="./assets/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="./assets/logo_light.png" alt="SDForge Logo" height="200">
  </picture>
</p>

## About

SDF Forge is a Python library for creating 3D models using Signed Distance Functions (SDFs). It provides a real-time, interactive rendering experience in a native desktop window, powered by GLSL raymarching.

## Features

- **Simple, Pythonic API:** Define complex shapes by combining primitives using standard operators (`|`, `-`, `&`) and chaining transformations.
- **Real-time Rendering with Hot-Reloading:** Get instant visual feedback in a lightweight native window powered by `moderngl` and `glfw`. Changes to your script are reloaded automatically.
- **Mesh Exporting:** Save your creations as `.stl`, `.obj`, or `.glb` files for 3D printing or use in other software.
- **Interactive Parameters:** Define parameters that can be controlled by UI sliders in real-time (with a compatible UI renderer).
- **Standalone GLSL Export:** Export your entire scene to a single, self-contained GLSL fragment shader for use in game engines or graphics applications.
- **Custom GLSL Primitives:** Write custom SDF logic directly in GLSL using the `Forge` object for maximum flexibility and performance.

## Getting Started

### Hello Forge

Define a simple shape and open a real-time preview window with just a few lines of code.

```python
from sdforge import sphere, box

# A sphere intersected with a box
shape = sphere(1) & box(1.5)

# Render a preview in a native window.
# An interactive orbit camera is used by default.
shape.render()
```

### Live Preview & Hot-Reloading
For an interactive workflow, wrap your scene definition in a `main()` function. When you save any changes to the file, the preview window will automatically update.

```python
from sdforge import box, sphere

def main():
    """
    While this script is running, try changing the values below and
    saving the file. The render window will update automatically.
    """
    box_size = 1.5
    sphere_radius = 1.2
    
    scene = box(box_size, radius=0.1) | sphere(sphere_radius)
    return scene

if __name__ == "__main__":
    # The render() function automatically enables hot-reloading by default
    # and looks for a `main` function in this file to call upon reload.
    scene = main()
    scene.render()
```

### Saving to a Mesh File
You can save any static model to an `.stl`, `.obj`, or `.glb` file. The `.save()` method uses the Marching Cubes algorithm to generate a mesh from the SDF.

```python
from sdforge import sphere, box

# A box with a sphere carved out of it
shape = box(1.5) - sphere(1.2)

# Save the model. Higher samples = more detail.
shape.save('model.obj', samples=2**22)
```

## Core Concepts

### Primitives & Operations
Create complex objects by starting with primitives and combining them with Python's bitwise operators: `|` for union, `&` for intersection, and `-` for difference.

```python
from sdforge import sphere, box

# A box with a sphere carved out of it.
b = box(size=1.5)
s = sphere(r=1.0)
scene = b - s

scene.render()
```

### Transformations & Shaping
Chain methods to transform, shape, and repeat objects.

```python
import numpy as np
from sdforge import box, Y

# A tall box
b = box(size=(0.5, 2.5, 0.5))

# Twist it around the Y-axis
# The 'k' parameter controls the amount of twist.
scene = b.twist(k=3.0).rotate(Y, np.pi / 4)

scene.render()
```

### Materials
You can assign a unique color to any object or group of objects using the `.color()` method.

```python
from sdforge import sphere, box

# A blue sphere is subtracted from a red box.
red_box = box(1.5, radius=0.1).color(1.0, 0.2, 0.2)
blue_sphere = sphere(1.2).color(0.3, 0.5, 1.0)
    
scene = red_box - blue_sphere

scene.render()
```

### Camera & Lighting
You can override the default interactive camera and lighting to set a static viewpoint or create specific lighting conditions by passing `Camera` and `Light` objects to the renderer.

```python
from sdforge import box, sphere, Camera, Light

def main():
    scene = box(1.5, radius=0.1) | sphere(1.2)
    
    # A camera positioned at (4, 3, 4), looking at the origin.
    cam = Camera(position=(4, 3, 4), target=(0, 0, 0), zoom=1.5)
    
    # A light source with soft shadows
    light = Light(position=(4, 5, 3), shadow_softness=16.0)

    return scene, cam, light

if __name__ == '__main__':
    scene, cam, light = main()
    scene.render(camera=cam, light=light)
```

## Advanced Usage

### Interactive Parameters
Use `Param` objects to define interactive, real-time parameters for your model. A compatible UI-enabled renderer would show these as sliders. The default renderer will use their default values.

```python
from sdforge import box, Param

# Create Param objects to control different aspects of the scene.
# Param(name, default_value, min_value, max_value)
p_size = Param("Box Size", 0.8, 0.2, 2.0)
p_radius = Param("Corner Radius", 0.1, 0.0, 0.5)

# Use the Param objects just like regular numbers.
scene = box(size=p_size, radius=p_radius)

scene.render()
```

### Custom GLSL with `Forge`
For complex or highly-performant shapes, you can write GLSL code directly. This object integrates perfectly with the rest of the API.

```python
from sdforge import sphere, Forge

# A standard library primitive
s = sphere(1.2)

# A custom shape defined with GLSL
# 'p' is the vec3 point in space
custom_box = Forge("""
    vec3 q = abs(p) - vec3(0.8);
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
""")

# The Forge object can be combined like any other shape
scene = s - custom_box

scene.render()
```

### Standalone GLSL Export
Generate a complete, self-contained GLSL fragment shader for your scene. This file can be used directly in applications like Godot, TouchDesigner, or Three.js.

```python
from sdforge import box, sphere

scene = box(1.5, radius=0.1) - sphere(1.2)

# This single call generates the entire shader file.
scene.export_shader("exported_shader.glsl")
```

## Installation

### 1. System Dependencies

The live viewer relies on `glfw`, which may require you to install its system-level libraries first. This is a common first step before installing the Python package.

**On Debian/Ubuntu:**
```bash
sudo apt-get install libglfw3-dev
```

**On macOS (with Homebrew):**
```bash
brew install glfw
```
**On Windows:**
`glfw` is generally bundled with the Python wheels, so no separate installation is typically needed.

### 2. Python Package

The library and its core dependencies can be installed using pip.

**Standard Installation:**
This will install the core library, including `numpy`, `moderngl`, and `glfw`, enabling all fundamental features like modeling, live preview with hot-reloading, and exporting to `.stl` and `.obj` formats.

```bash
pip install sdforge
```

**Optional Features:**
For additional functionality, you can install "extras":

*   **`.glb` Export:** To enable saving models to the GLB format, a modern and efficient format for web and game engines.
    ```bash
    pip install "sdforge[export]"
    ```

*   **Interactive UI:** To enable UI widgets like sliders for `Param` objects.
    ```bash
    pip install "sdforge[ui]"
    ```

*   **Full Installation:** To install all optional features at once.
    ```bash
    pip install "sdforge[full]"
    ```

## Acknowledgements

This project is inspired by the simplicity and elegant API of Michael Fogleman's [fogleman/sdf](https://github.com/fogleman/sdf) library. SDF Forge aims to build on that foundation by adding a real-time, interactive GLSL-powered renderer.
