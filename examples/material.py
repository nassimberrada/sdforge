import sys
from sdforge import box, sphere

def main():
    """
    Demonstrates applying different colors to objects.
    
    The renderer automatically collects these colors and makes them available
    to the shader.
    """
    print("--- SDForge Material & Color Example ---")

    # A blue sphere is subtracted from a red box.
    red_box = box(1.5, radius=0.1).color(1.0, 0.2, 0.2)
    blue_sphere = sphere(1.2).color(0.3, 0.5, 1.0)
    
    scene = red_box - blue_sphere
    
    return scene

if __name__ == "__main__":
    scene = main()
    scene.render()