import pytest
import numpy as np
from sdforge import SDFNode, sphere, box, torus, line, cylinder, cone, plane, octahedron, ellipsoid, circle, rectangle
from sdforge.render import SceneCompiler
from tests.conftest import requires_glsl_validator

# Import example functions and their expected return types for parametrization
from examples.primitives import (
    sphere_example,
    box_example,
    cylinder_example,
    torus_example,
    cone_example,
)
from sdforge.api.primitives import Sphere, Box, Cylinder, Torus, Cone

# --- API and Callable Tests ---

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

def test_box_api():
    b1 = box()
    assert np.allclose(b1.size, (1,1,1))
    b2 = box(size=2.0)
    assert np.allclose(b2.size, (2,2,2))
    b3 = box(size=(1,2,3), radius=0.1)
    assert np.allclose(b3.size, (1,2,3))
    assert b3.radius == 0.1
    b4 = box(x=1,y=2,z=3)
    assert np.allclose(b4.size, (1,2,3))

def test_box_callable():
    b_callable = box(size=2.0).to_callable()
    points = np.array([[0, 0, 0], [1.5, 0, 0], [1, 1, 1], [2, 2, 0]])
    expected = np.array([-1.0, 0.5, 0.0, np.sqrt(1**2 + 1**2)])
    assert np.allclose(b_callable(points), expected)

def test_torus_api():
    t = torus(major=2.0, minor=0.5)
    assert t.major == 2.0
    assert t.minor == 0.5
    
def test_torus_callable():
    t_callable = torus(major=1.0, minor=0.2).to_callable()
    points = np.array([[1, 0, 0], [1, 0.2, 0], [1, 0.3, 0], [0, 0, 0]])
    expected = np.array([-0.2, 0.0, 0.1, 0.8])
    assert np.allclose(t_callable(points), expected)

def test_line_api():
    l1 = line(a=(0,0,0), b=(1,1,1), radius=0.2, rounded_caps=True)
    assert l1.rounded_caps is True
    l2 = line(a=(0,0,0), b=(1,1,1), radius=0.2, rounded_caps=False)
    assert l2.rounded_caps is False
    
def test_cylinder_callable():
    c_callable = cylinder(radius=1.0, height=2.0).to_callable()
    points = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0], [1.5,0,0]])
    expected = np.array([-1.0, 0.0, 0.0, 0.0, 0.5])
    assert np.allclose(c_callable(points), expected)

def test_plane_callable():
    p_callable = plane(normal=(0,1,0), offset=0.5).to_callable()
    points = np.array([[10, -0.5, 0], [10, 0.5, 0], [10, -1.5, 0]])
    expected = np.array([0.0, 1.0, -1.0])
    assert np.allclose(p_callable(points), expected)

def test_octahedron_callable():
    o_callable = octahedron(size=1.0).to_callable()
    points = np.array([[0,0,0], [1,0,0], [0.5,0.5,0]])
    expected = np.array([-1.0, 0.0, 0.0]) * 0.57735027
    assert np.allclose(o_callable(points), expected)

def test_ellipsoid_callable():
    e_callable = ellipsoid(radii=(1,2,3)).to_callable()
    points = np.array([[1,0,0], [0,2,0], [0,0,3]])
    assert np.allclose(e_callable(points), 0.0, atol=1e-6)


# --- Equivalence and Compilation Tests ---

PRIMITIVE_TEST_CASES = [
    sphere(r=1.2),
    box(size=(1.0, 0.5, 0.8)),
    box(size=1.2, radius=0.1),
    torus(major=1.0, minor=0.25),
    line(a=(0,0,0), b=(0,1,0), radius=0.1, rounded_caps=True),
    line(a=(0,0,0), b=(0,1,0), radius=0.1, rounded_caps=False),
    cylinder(radius=0.5, height=1.5),
    cylinder(radius=0.5, height=1.5, round_radius=0.1),
    cone(height=1.2, radius1=0.6, radius2=0.2),
    cone(height=1.2, radius1=0.6, radius2=0.0),
    plane(normal=(0.6, 0.8, 0), offset=0.5),
    octahedron(size=1.3),
    ellipsoid(radii=(1.0, 0.5, 0.7)),
    circle(r=1.5),
    rectangle(size=(1.0, 0.5))
]

@pytest.mark.usefixtures("assert_equivalence")
@pytest.mark.parametrize("sdf_obj", PRIMITIVE_TEST_CASES, ids=[type(p).__name__ for p in PRIMITIVE_TEST_CASES])
def test_primitive_equivalence(assert_equivalence, sdf_obj):
    """Tests numeric equivalence between Python and GLSL for all primitives."""
    assert_equivalence(sdf_obj)

@requires_glsl_validator
@pytest.mark.parametrize("sdf_obj", PRIMITIVE_TEST_CASES, ids=[type(p).__name__ for p in PRIMITIVE_TEST_CASES])
def test_primitive_glsl_compiles(validate_glsl, sdf_obj):
    """Tests that the GLSL generated for all primitives is syntactically valid."""
    scene_code = SceneCompiler().compile(sdf_obj)
    validate_glsl(scene_code)


# --- Example File Tests ---

EXAMPLE_TEST_CASES = [
    (sphere_example, Sphere),
    (box_example, Box),
    (cylinder_example, Cylinder),
    (torus_example, Torus),
    (cone_example, Cone),
]

@pytest.mark.parametrize("example_func, expected_class", EXAMPLE_TEST_CASES, ids=[f[0].__name__ for f in EXAMPLE_TEST_CASES])
def test_primitive_example_runs(example_func, expected_class):
    """
    Tests that the primitive example functions from the examples file
    run without errors and return a valid SDFNode of the correct type.
    """
    scene = example_func()
    assert isinstance(scene, SDFNode)
    assert isinstance(scene, expected_class)