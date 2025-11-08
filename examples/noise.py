import sys
from sdforge import sphere, box

def noise_displacement_example():
    """Applies procedural noise to the surface of a sphere."""
    s = sphere(r=1.2)
    # Higher scale = finer detail, higher strength = more displacement.
    return s.displace_by_noise(scale=8.0, strength=0.1)

def sine_wave_displacement_example():
    """Applies a custom GLSL sine wave displacement to a box."""
    b = box(size=1.8, radius=0.1)
    # You can use any GLSL expression that returns a float.
    # The variable 'p' represents the point in space being sampled.
    glsl_code = "sin(p.x * 20.0) * sin(p.z * 20.0) * 0.05"
    return b.displace(glsl_code)

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Noise & Displacement Examples ---")
    
    examples = {
        "noise": noise_displacement_example,
        "sine_wave": sine_wave_displacement_example,
    }
    
    if len(sys.argv) < 2:
        print("\nPlease provide the name of an example to run.")
        print("Available examples:")
        for key in examples:
            print(f"  - {key}")
        print(f"\nUsage: python {sys.argv[0]} <example_name>")
        return

    example_name = sys.argv[1]
    scene_func = examples.get(example_name)
    
    if not scene_func:
        print(f"\nError: Example '{example_name}' not found.")
        print("Available examples are:")
        for key in examples:
            print(f"  - {key}")
        return

    print(f"Rendering: {example_name.replace('_', ' ').title()} Example")
    result = scene_func()
    if isinstance(result, tuple):
        scene, cam = result
        scene.render(camera=cam)
    else:
        scene = result
        scene.render()

if __name__ == "__main__":
    main()