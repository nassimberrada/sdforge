import sys
import numpy as np
from sdforge import sphere, box, cylinder, X, Y, Param, Group

def union_example(): return sphere(0.8) | box((1.5, 0.5, 0.5))
def intersection_example(): return sphere(1.0).translate((-0.5, 0, 0)) & sphere(1.0).translate((0.5, 0, 0))
def difference_example(): return box(1.5) - sphere(1.0)

def smooth_union_example():
    s1 = sphere(0.7).translate((-0.5, 0, 0))
    s2 = sphere(0.7).translate((0.5, 0, 0))
    return s1.union(s2, blend=0.3)

def masked_blend_example():
    s1 = sphere(0.7).translate((-0.6, 0, 0))
    s2 = sphere(0.7).translate((0.6, 0, 0))
    mask_obj = box((3, 2, 3)).translate((0, 1.0, 0))
    return s1.union(s2, blend=0.4, mask=mask_obj, mask_falloff=0.2)

def fillet_difference_example():
    plate = box((2.0, 0.5, 2.0))
    hole = cylinder(0.5, 1.0)
    return plate.difference(hole, blend=0.1)

def linear_difference_example():
    plate = box((2.0, 0.5, 2.0))
    hole = cylinder(0.5, 1.0)
    return plate.difference(hole, blend=0.1, blend_type='linear')

def morphing_example():
    p_morph = Param("Morph Factor", 0.5, 0.0, 1.0)
    return sphere(1.0).morph(box(1.5), factor=p_morph)

def masked_morph_example():
    mask_obj = box(3.0).translate((1.5, 0, 0))
    return sphere(1.0).morph(box(1.5), factor=1.0, mask=mask_obj, mask_falloff=0.5)

def group_transform_example():
    b = box((1.5, 0.5, 0.5)).round(0.1).translate((-1, 0, 0))
    s = sphere(0.5).translate((1, 0, 0))
    g = Group(b, s)
    return g.rotate(Y, np.pi / 4)

def main():
    print("--- SDForge Composition Examples ---")
    examples = {
        "union": union_example, "intersection": intersection_example, "difference": difference_example,
        "smooth_union": smooth_union_example, "masked_blend": masked_blend_example,
        "fillet_cut": fillet_difference_example, "linear_cut": linear_difference_example,
        "morph": morphing_example, "masked_morph": masked_morph_example,
        "group": group_transform_example,
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