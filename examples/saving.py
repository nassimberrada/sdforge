import sys
import os
from sdforge import box, sphere

def auto_bounds_save_example():
    """
    Creates a simple object and saves it to a file.
    The '.save()' method will automatically estimate the necessary bounds.
    """
    scene = box(1.5, radius=0.1) | sphere(1.2)
    
    output_path = "auto_bounds_model.stl"
    print(f"\nSaving model to '{output_path}' with automatic bounds estimation...")
    
    # We can control the mesh density with the 'samples' parameter.
    # Higher is more detailed but slower.
    scene.save(output_path, samples=2**20)
    
    if os.path.exists(output_path):
        print(f"To view the model, run: meshlab {output_path}")
    else:
        print("Error: Model saving failed.")

def manual_bounds_save_example():
    """
    Creates an object and saves it with manually specified bounds and resolution.
    This gives more control but requires knowing the object's size.
    """
    scene = box(1.5, radius=0.1) | sphere(1.2)
    
    output_path = "manual_bounds_model.obj"
    print(f"\nSaving model to '{output_path}' with manual bounds...")
    
    # Define the volume to mesh, e.g., a 4x4x4 cube around the origin.
    bounds = ((-2, -2, -2), (2, 2, 2))
    
    # Use a higher sample count for a more detailed mesh.
    scene.save(output_path, bounds=bounds, samples=2**22)

    if os.path.exists(output_path):
        print(f"To view the model, run: meshlab {output_path}")
    else:
        print("Error: Model saving failed.")


def main():
    """
    Runs a saving example based on a command-line argument.
    """
    print("--- SDForge Mesh Saving Examples ---")
    
    examples = {
        "auto": auto_bounds_save_example,
        "manual": manual_bounds_save_example,
    }
    
    if len(sys.argv) < 2:
        print("\nPlease provide the name of an example to run.")
        print("Available examples:")
        for key in examples:
            print(f"  - {key}")
        print(f"\nUsage: python {sys.argv[0]} <example_name>")
        return

    example_name = sys.argv[1]
    example_func = examples.get(example_name)
    
    if not example_func:
        print(f"\nError: Example '{example_name}' not found.")
        print("Available examples are:")
        for key in examples:
            print(f"  - {key}")
        return

    # Just call the function, it handles its own logic and printing.
    example_func()

if __name__ == "__main__":
    main()