import sys
import numpy as np
from sdforge import box, sphere, circle, rectangle, cylinder, X, Y, Z

def translation_example(): return sphere(0.6) | (sphere(0.6) + (X * 1.5))
def scale_example(): b = box(1.0); return b | b.scale((0.5, 2.0, 0.5)).translate(Y * 1.5)
def rotation_example(): b = box((1.5, 0.8, 0.3)); return b | b.rotate(Z, np.pi/3).rotate(X, np.pi/6)
def rotation_axis_example(): return box((2.0, 0.2, 0.5)).rotate(np.array([1.0, 1.0, 0.0]), np.pi/4)
def orientation_example(): b = box((1.5, 0.8, 0.3)); return b | b.orient('x').translate(Y * 1.2)

def twist_example(): return box((0.5, 2.5, 0.5)).twist(strength=3.0)
def masked_twist_example():
    b = box((0.5, 2.5, 0.5))
    mask = box((1.0, 1.5, 1.0)).translate((0, 1.0, 0))
    return b.twist(strength=5.0, mask=mask, mask_falloff=0.0)

def bend_example(): return box((3.0, 0.4, 0.8)).bend(Y, curvature=0.5)
def warp_example():
    s = sphere(1.0)
    return s.warp(frequency=2.0, strength=0.5)

def repeat_example(): return sphere(0.4).translate(X * 0.8).repeat((2.0, 2.0, 0.0))
def limited_repeat_example(): return sphere(0.4).limited_repeat(spacing=(1.2, 0, 0), limits=(2, 0, 0))
def polar_repeat_example(): return box((0.8, 0.4, 0.2)).round(0.05).translate(X * 1.2).polar_repeat(8)
def mirror_example(): return box((1.0, 0.5, 0.5)).round(0.1).translate((0.8, 0.5, 0)).mirror(X | Y)

def round_example(): return box((1.5, 1.0, 0.5)).round(0.2)
def masked_round_example():
    b = box(1.5)
    mask = sphere(0.8).translate((0.75, 0.75, 0.75))
    return b.round(0.3, mask=mask, mask_falloff=0.1)

def shell_example(): return sphere(1.0).shell(0.1)
def masked_shell_example():
    s = sphere(1.0)
    mask = box(2.0).translate((0, 1.0, 0))
    return s.shell(0.1, mask=mask, mask_falloff=0.2)

def extrude_example(): return circle(0.8).extrude(1.5)
def revolve_example():
    r1 = rectangle((0.4, 1.0)).translate((0.7, 0, 0))
    r2 = rectangle((0.8, 0.2)).translate((0.5, 0, 0))
    return (r1 | r2).revolve()

def noise_displacement_example(): return sphere(1.2).displace_by_noise(scale=8.0, strength=0.1)
def sine_wave_displacement_example(): return box(1.8).round(0.1).displace("sin(p.x * 20.0) * sin(p.z * 20.0) * 0.05")

def simple_material_example(): return box(1.5).round(0.1).color(1.0, 0.2, 0.2) - sphere(1.2).color(0.3, 0.5, 1.0)
def masked_material_example():
    base = box(2.0).color(0.9, 0.9, 0.9)
    mask_shape = cylinder(0.5, 3.0).rotate(X, 1.57).translate((0, 0.5, 0))
    return base.color(1.0, 0.1, 0.1, mask=mask_shape)

def main():
    print("--- SDForge Operator Examples ---")
    examples = {
        "translation": translation_example, "scale": scale_example, "rotation": rotation_example,
        "rotation_axis": rotation_axis_example, "orientation": orientation_example,
        "twist": twist_example, "masked_twist": masked_twist_example, "bend": bend_example,
        "warp": warp_example, "repeat": repeat_example, "limited_repeat": limited_repeat_example,
        "polar_repeat": polar_repeat_example, "mirror": mirror_example,
        "round": round_example, "masked_round": masked_round_example,
        "shell": shell_example, "masked_shell": masked_shell_example,
        "extrude": extrude_example, "revolve": revolve_example,
        "noise": noise_displacement_example, "sine_wave": sine_wave_displacement_example,
        "mat_simple": simple_material_example, "mat_masked": masked_material_example,
    }

    if len(sys.argv) < 2:
        print("Available examples:", ", ".join(examples.keys()))
        return

    func = examples.get(sys.argv[1])
    if func:
        scene = func()
        scene.render()
    else: print(f"Example '{sys.argv[1]}' not found.")

if __name__ == "__main__":
    main()