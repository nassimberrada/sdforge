from sdforge import *

def main():
    """
    Demonstrates grouping objects to transform them together.

    This example shows how to:
    - Create multiple objects.
    - Combine them into a `Group`.
    - Apply a single transformation (like `.translate()` or `.rotate()`) to the
      entire group, affecting all children simultaneously.
    """
    # Create components for a simple face
    head = sphere(1.0).color(1.0, 0.8, 0.2)
    eye = sphere(0.2).color(0.1, 0.1, 0.1)
    
    # Position the eyes
    left_eye = eye.translate(-X * 0.4 + Y * 0.2)
    right_eye = eye.translate(X * 0.4 + Y * 0.2)

    # Combine all parts into a group
    face_group = Group(head, left_eye, right_eye)

    # Now, any transform applied to the group affects all its children.
    # Let's make two faces and move them around.
    face1 = face_group.translate(-X * 1.5)
    face2 = face_group.translate(X * 1.5).rotate(Z, 0.5)

    return face1 | face2

if __name__ == "__main__":
    model = main()
    if model:
        model.render(watch=True)