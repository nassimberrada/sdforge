from sdforge import box, sphere

def main():
    """
    A simple scene that can be modified live.
    
    While this script is running, try changing the values below and
    saving the file. The render window will update automatically.
    """
    
    # Try changing this value from 1.5 to 0.8 and save the file.
    box_size = 1.5
    
    # Try changing this value from 1.2 to 2.0 and save the file.
    sphere_radius = 1.2
    
    # Try uncommenting the line below and save the file.
    # sphere_radius = 0.5
    
    scene = box(box_size, radius=0.1) | sphere(sphere_radius)
    
    print(f"Reloaded: Box Size = {box_size}, Sphere Radius = {sphere_radius}")
    
    return scene

if __name__ == "__main__":
    # The render() function automatically enables hot-reloading by default.
    # It looks for a `main` function in this file to call upon reload.
    scene = main()
    scene.render()