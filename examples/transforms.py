import sys
import numpy as np
from sdforge import box, sphere, X, Y, Z, Param, Group

def translation_example():
    """Shows a shape and its translated copy, joined by a union."""
    s = sphere(radius=0.6)
    return s | (s + (X * 1.5))

def scale_example():
    """Shows a shape and its non-uniformly scaled copy, joined by a union."""
    b = box(size=1.0)
    return b | b.scale((0.5, 2.0, 0.5)).translate(Y * 1.5)

def rotation_example():
    """Shows a shape and its rotated copy, joined by a union."""
    b = box(size=(1.5, 0.8, 0.3))
    return b | b.rotate(Z, np.pi / 3).rotate(X, np.pi / 6)

def rotation_axis_example():
    """Shows rotation around an arbitrary axis."""
    b = box(size=(2.0, 0.2, 0.5))
    axis = np.array([1.0, 1.0, 0.0])
    return b.rotate(axis, np.pi / 4)

def orientation_example():
    """Shows how .orient() can re-orient a shape along a new axis."""
    b = box(size=(1.5, 0.8, 0.3))
    return b | b.orient('x').translate(Y * 1.2)

def twist_example():
    """Demonstrates twisting a tall shape around its Y-axis."""
    b = box(size=(0.5, 2.5, 0.5))
    return b.twist(strength=3.0)

def masked_twist_example():
    """
    Demonstrates applying a twist only to the top half of a box.
    """
    b = box(size=(0.5, 2.5, 0.5))    
    mask = box(size=(1.0, 1.5, 1.0)).translate((0, 1.0, 0))    
    return b.twist(strength=5.0, mask=mask, mask_falloff=0.0)

def bend_example():
    """Demonstrates bending a long shape into an arc."""
    plank = box(size=(3.0, 0.4, 0.8))
    return plank.bend(Y, curvature=0.5)

def warp_sphere_example():
    """Applies domain warping to a sphere."""
    p_freq = Param("Frequency", 2.0, 0.5, 5.0)
    p_strength = Param("Strength", 0.5, 0.0, 2.0)
    s = sphere(radius=1.0)
    return s.warp(frequency=p_freq, strength=p_strength)

def warp_box_example():
    """Applies domain warping to a box."""
    b = box(size=1.5)
    return b.warp(frequency=1.5, strength=0.3)

def warp_comparison_example():
    """Compares Twist, Displacement, and Warping side-by-side."""
    twisted = box(1.0).twist(strength=2.0).translate((-2.5, 0, 0))
    displaced = box(1.0).displace_by_noise(scale=2.0, strength=0.2)
    warped = box(1.0).warp(frequency=2.0, strength=0.4).translate((2.5, 0, 0))
    return Group(twisted, displaced, warped)

def repeat_example():
    """Shows infinite repetition of a shape."""
    s = sphere(radius=0.4).translate(X * 0.8)
    return s.repeat(spacing=(2.0, 2.0, 0.0))

def limited_repeat_example():
    """Shows finite repetition of a shape."""
    s = sphere(radius=0.4)
    return s.repeat(spacing=(1.2, 0, 0), count=(2, 0, 0))

def polar_repeat_example():
    """Repeats a shape in a circle around the Y-axis."""
    b = box(size=(0.8, 0.4, 0.2)).round(0.05).translate(X * 1.2)
    return b.repeat(count=8)

def mirror_example():
    """Creates symmetry by mirroring a shape across axes."""
    b = box(size=(1.0, 0.5, 0.5)).round(0.1).translate((0.8, 0.5, 0))
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
        "rotation_axis": rotation_axis_example,
        "orientation": orientation_example,
        "twist": twist_example,
        "masked_twist": masked_twist_example,
        "bend": bend_example,
        "warp_sphere": warp_sphere_example,
        "warp_box": warp_box_example,
        "warp_compare": warp_comparison_example,
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