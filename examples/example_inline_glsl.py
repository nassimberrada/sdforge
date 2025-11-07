from sdforge import *

def main():
    """
    This function defines the SDF model.
    For hot-reloading to work, it must return the final SDF object.
    """
    # A standard library primitive
    s = box(1.2)
    # A custom shape defined with GLSL
    # 'p' is the vec3 point in space
    model = Forge("""
        float k = 10.0;
        float c = cos(k*p.y);
        float s = sin(k*p.y);
        mat2  m = mat2(c,-s,s,c);
        vec3  q = vec3(m*p.xz,p.y);
        return length(q) - 1.5;
    """)

    # The original operation `s - model` results in an empty shape because
    # the box is entirely contained within the custom model.
    # Subtracting the box from the model produces a visible result.
    f = model - s
    
    # Return the final object for the renderer
    return f

# This is the standard way to run the script.
if __name__ == '__main__':
    sdf_object = main()
    if sdf_object:
        # The render call is now a simple, blocking function call.
        # It will run until you close the window.
        sdf_object.render(watch=True)