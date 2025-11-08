import numpy as np
from sdforge import SDFNode, sphere
from sdforge.engine import SceneCompiler
from tests.conftest import requires_glsl_validator
from examples.primitives import sphere_example

def test_sphere_api():
    """Tests basic API usage of the sphere primitive."""
    s1 = sphere()
    assert isinstance(s1, SDFNode)
    assert s1.r == 1.0

    s2 = sphere(r=1.5)
    assert s2.r == 1.5

def test_sphere_callable():
    """Tests the numeric accuracy of the sphere's Python callable."""
    s_callable = sphere(r=1.0).to_callable()
    points = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]])
    expected = np.array([-1.0, -0.5, 0.0, 1.0])
    assert np.allclose(s_callable(points), expected)

def test_sphere_equivalence(assert_equivalence):
    """Tests numeric equivalence between the Python and GLSL implementations."""
    assert_equivalence(sphere(r=1.2))

@requires_glsl_validator
def test_sphere_glsl_compiles(validate_glsl):
    """
    Tests that the GLSL generated for a sphere is syntactically valid.
    """
    s = sphere(r=1.0)
    scene_code = SceneCompiler().compile(s)
    validate_glsl(scene_code)

def test_sphere_example_runs():
    """
    Tests that the sphere_example() function from the examples file
    runs without errors and returns a valid SDFNode.
    """
    scene = sphere_example()    
    assert isinstance(scene, SDFNode)    
    from sdforge.api.primitives import Sphere
    assert isinstance(scene, Sphere)