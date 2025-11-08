import sys
from sdforge import Forge, box

def simple_forge_example():
    """
    A sphere created using a raw GLSL expression in a Forge object.
    """
    return Forge("length(p) - 1.0")

def uniform_forge_example():
    """
    A box created using a Forge object with a custom uniform to control its size.
    The renderer will automatically find and upload this uniform.
    """
    glsl_code = """
        vec3 q = abs(p) - u_size;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
    """
    # Uniforms are a dictionary of { 'glsl_name': python_value }
    uniforms = { 'u_size': 0.8 }
    return Forge(glsl_code, uniforms=uniforms)

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Custom GLSL (Forge) Examples ---")
    
    examples = {
        "simple": simple_forge_example,
        "uniform": uniform_forge_example,
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