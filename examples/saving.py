import sys
import os
from sdforge import box, sphere

def auto_bounds_save_example():
    """
    Creates a simple object and saves it to a file.
    The '.save()' method will automatically estimate the necessary bounds.
    """
    scene = box(1.5).round(0.1) | sphere(radius=1.2)
    
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
    scene = box(1.5).round(0.1) | sphere(radius=1.2)
    
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

def decimated_save_example():
    """
    Saves a mesh with post-process simplification to reduce triangle count.
    This requires the 'trimesh' library to be installed.
    """
    # A box has large flat faces, making it a good candidate for decimation.
    scene = box(size=(2, 1, 1.5)).round(0.1)
    
    output_path = "decimated_model.stl"
    print(f"\nSaving model to '{output_path}' with 90% triangle reduction...")
    
    # We generate a high-resolution mesh first.
    # Then, decimate_ratio=0.9 will aim to remove 90% of the triangles.
    scene.save(output_path, samples=2**22, decimate_ratio=0.9)
    
    if os.path.exists(output_path):
        print(f"To view the model, run: meshlab {output_path}")
        print("Notice how the flat surfaces have far fewer triangles than a normal export.")
    else:
        print("Error: Model saving failed.")

def adaptive_save_example():
    """
    Saves a mesh using adaptive octree subdivision.
    This is much faster and more memory-efficient for sparse or hollow objects,
    as it only evaluates points near the object's surface.
    """
    # A hollow box is a perfect example where adaptive meshing shines.
    # Uniform sampling would waste millions of points inside the hollow area.
    scene = box(2.0).round(0.1).shell(0.05)
    
    output_path = "adaptive_model.stl"
    print(f"\nSaving model to '{output_path}' with adaptive meshing...")
    
    # Instead of `samples`, we use `adaptive=True` and control detail
    # with `octree_depth`. A depth of 8 gives a 256x256x256 effective resolution.
    scene.save(output_path, adaptive=True, octree_depth=8)
    
    if os.path.exists(output_path):
        print(f"To view the model, run: meshlab {output_path}")
        print("This mesh was generated much faster than it would have been with uniform sampling.")
    else:
        print("Error: Model saving failed.")

def voxel_size_save_example():
    """
    Saves a mesh by specifying a physical resolution (voxel_size) instead of 
    an abstract sample count. This is useful for engineering applications.
    """
    scene = sphere(1.0).round(0.1)
    
    output_path = "voxel_size_model.stl"
    print(f"\nSaving model to '{output_path}' with 0.05 unit voxel size...")
    
    # By providing `voxel_size`, the saver automatically calculates
    # the required grid dimensions (or octree depth) to meet this resolution.
    scene.save(output_path, voxel_size=0.05, adaptive=True)
    
    if os.path.exists(output_path):
        print(f"To view the model, run: meshlab {output_path}")
    else:
        print("Error: Model saving failed.")

def dual_contouring_save_example():
    """
    Saves a mesh using the Dual Contouring algorithm.
    This algorithm is often better at preserving sharp features than
    Marching Cubes, at the cost of being slower.
    """
    # A box with no rounding is a good test case for sharp edges.
    scene = box(size=(1.5, 1.0, 1.0))

    output_path = "dual_contouring_model.stl"
    print(f"\nSaving model to '{output_path}' with Dual Contouring...")
    
    # We specify the algorithm in the save() call.
    scene.save(output_path, samples=2**18, algorithm='dual_contouring')
    
    if os.path.exists(output_path):
        print(f"To view the model, run: meshlab {output_path}")
        print("Notice how the sharp edges are preserved much better than a typical marching cubes export.")
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
        "decimate": decimated_save_example,
        "adaptive": adaptive_save_example,
        "voxel": voxel_size_save_example,
        "dual_contouring": dual_contouring_save_example,
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