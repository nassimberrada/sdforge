from sdforge import *

def main():
    """
    Demonstrates the use of per-object materials.
    """
    # Define a red sphere and a blue box
    s = sphere(1.2).color(1, 0.2, 0.2)
    b = box(1.5).color(0.2, 0.8, 1.0)

    # Union the two colored objects
    f = s | b.translate(X * 0.8)

    # A green cylinder for subtraction
    c = cylinder(0.5).color(0.2, 1.0, 0.3)
    
    # Subtract a twisted cylinder
    # Note: the color of the subtracted object is ignored,
    # but the color of the base object (f) is preserved.
    f -= c.twist(5.0)

    return f

if __name__ == "__main__":
    model = main()
    if model:
        # Render with a dark grey background
        model.render(watch=True, bg_color=(1, 1, 1))