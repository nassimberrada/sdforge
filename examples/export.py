import sys
import os
from sdforge import box, sphere, Param

def main():
    """
    Demonstrates exporting a complete, standalone GLSL shader file.
    """
    print("--- SDForge Shader Export Example ---")

    # Create a scene with some complexity and a parameter.
    p_radius = Param("Sphere Radius", 0.8, 0.5, 1.5)
    scene = box(1.5, radius=0.1) - sphere(p_radius)

    output_path = "exported_shader.glsl"
    print(f"Exporting scene to '{output_path}'...")
    
    # This single call generates the entire shader file.
    scene.export_shader(output_path)
    
    if os.path.exists(output_path):
        print("\nSUCCESS! You can now use this shader file in other applications")
        print("that support GLSL fragment shaders, like Godot, TouchDesigner, or Three.js.")
    else:
        print("\nERROR: Shader export failed.")

    # We can also render the scene as usual.
    return scene

if __name__ == "__main__":
    scene = main()
    scene.render()