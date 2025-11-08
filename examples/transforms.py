import sys
import numpy as np
from sdforge import box, sphere, X, Y, Z

def translation_example():
    """Shows a shape and its translated copy, joined by a union."""
    s = sphere(r=0.6)
    # A union of the original sphere and a translated copy.
    # The '+' operator is a shortcut for .translate()
    return s | (s + (X * 1.5))

def scale_example():
    """Shows a shape and its non-uniformly scaled copy, joined by a union."""
    b = box(size=1.0)
    # A union of the original box and a scaled/translated copy.
    # The '*' operator is a shortcut for uniform scaling.
    return b | b.scale((0.5, 2.0, 0.5)).translate(Y * 1.5)

def rotation_example():
    """Shows a shape and its rotated copy, joined by a union."""
    # A non-symmetrical box makes rotation obvious.
    b = box(size=(1.5, 0.8, 0.3))
    return b | b.rotate(Z, np.pi / 3).rotate(X, np.pi / 6)

def orientation_example():
    """Shows how .orient() can re-orient a shape along a new axis."""
    # A non-symmetrical box is used to make the orientation change clear.
    b = box(size=(1.5, 0.8, 0.3))
    # The original shape is Z-aligned by default.
    # .orient('x') re-orients it so its longest side is along the X-axis.
    return b | b.orient('x').translate(Y * 1.2)

def twist_example():
    """Demonstrates twisting a tall shape around its Y-axis."""
    b = box(size=(0.5, 2.5, 0.5))
    # The 'k' parameter controls the amount of twist in radians per unit of height.
    return b.twist(k=3.0)

def bend_example():
    """Demonstrates bending a long shape into an arc."""
    # A long, thin box is a good shape to visualize bending.
    plank = box(size=(3.0, 0.4, 0.8))
    # We bend it around the Y-axis. The 'k' parameter controls the curvature.
    return plank.bend(Y, k=0.5)

def repeat_example():
    """Shows infinite repetition of a shape."""
    # Start with a single sphere, offset from the origin.
    s = sphere(r=0.4).translate(X * 0.8)
    # Repeat it infinitely along the X and Y axes with a spacing of 2.0 units.
    # The Z spacing of 0 means no repetition along that axis.
    return s.repeat((2.0, 2.0, 0.0))

def limited_repeat_example():
    """Shows finite repetition of a shape."""
    s = sphere(r=0.4)
    # Repeat the sphere with a spacing of 1.2 units along the X-axis.
    # The limits=(2, 0, 0) means it will repeat twice in the positive direction
    # and twice in the negative direction, for a total of 5 spheres.
    return s.limited_repeat(spacing=(1.2, 0, 0), limits=(2, 0, 0))

def polar_repeat_example():
    """Repeats a shape in a circle around the Y-axis."""
    # Start with a shape that is offset from the origin (the axis of rotation).
    b = box(size=(0.8, 0.4, 0.2), radius=0.05).translate(X * 1.2)
    # Repeat it 8 times.
    return b.polar_repeat(8)

def mirror_example():
    """Creates symmetry by mirroring a shape across axes."""
    # Start with one quarter of a shape, offset into one quadrant.
    b = box(size=(1.0, 0.5, 0.5), radius=0.1).translate((0.8, 0.5, 0))
    # Mirror across the X and Y axes to create a symmetrical object.
    # (X | Y) is equivalent to (1, 1, 0)
    return b.mirror(X | Y)

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Transform Examples ---")
    
    examples = {
        "translation": translation_example,
        "scale": scale_example,
        "rotation": rotation_example,
        "orientation": orientation_example,
        "twist": twist_example,
        "bend": bend_example,
        "repeat": repeat_example,
        "limited_repeat": limited_repeat_example,
        "polar_repeat": polar_repeat_example,
        "mirror": mirror_example,
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