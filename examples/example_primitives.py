from sdforge import *

def main():
    """
    Demonstrates the expanded library of SDF primitives.
    """
    # A ground plane
    ground = plane(normal=Y, offset=0)
    
    # A cone acting as a tree top
    tree_top = cone(height=1.5, radius1=0.8).translate(Y * 1.5)

    # A capsule acting as the tree trunk
    trunk = line(Y * 0, Y * 1.5, radius=0.2)

    tree = tree_top | trunk

    # An octahedron "gem"
    gem = octahedron(0.5).translate(-X * 2 + Y * 0.5)

    # A rounded box "pedestal" for the gem
    pedestal = box(size=(0.7, 0.2, 0.7), radius=0.05).translate(-X * 2 + Y * 0.1)

    # A hex prism "nut"
    nut = hex_prism(radius=0.4, height=0.2).translate(X * 2 + Y * 0.2)
    # Cut a hole through the nut
    nut -= cylinder(radius=0.2)

    # An ellipsoid "stone"
    stone = ellipsoid((0.8, 0.4, 0.5)).translate(Y * 0.4 + X * 3.5)

    # A pyramid
    pyr = pyramid(height=1.0).translate(-X * 4 + Y * 0.5)

    # A rounded cone
    rc = round_cone(radius1=0.6, radius2=0.2, height=1.2).translate(-X * 5.5 + Y * 0.6)

    # A box frame
    bf = box_frame(size=(1, 1, 1), edge_radius=0.1).translate(X * 5.5 + Y * 0.6)

    # Combine all the objects into one scene
    scene = tree | gem | pedestal | nut | stone | pyr | rc | bf

    # Use a boolean operation with the ground plane to cut everything off at y=0
    final_scene = scene & ground

    return final_scene

if __name__ == "__main__":
    model = main()
    if model:
        model.render(watch=True)