# SDF Forge - GLSL Raymarching Toolbox

SDF Forge is a collection of GLSL functions used to setup a 3D SDF raymarching scene.

## File Structure

The GLSL library is organized into the following directories:

```
sdforge/
├── sdf/
│   ├── operations.glsl  # Functions to combine and manipulate shapes
│   └── primitives.glsl  # Basic SDF shapes (sphere, cube, etc.)
├── scene/
│   ├── camera.glsl      # Camera controllers (orbit, fixed, etc.)
│   ├── lighting.glsl    # Lighting, shadows, and ambient occlusion
│   └── raymarching.glsl # Core raymarching logic and normal estimation
│   └── render.glsl      # Core rendering pipeline
└── utils/
    ├── math.glsl        # General mathematical utility functions
    └── noise.glsl       # Noise functions (random, fbm, etc.)
```

`starter.html` is a reusable starter template for rendering a basic raymarching scene.

## Getting Started

Just clone or download the repository to your local machine and use the functions in your raymarching projects.
