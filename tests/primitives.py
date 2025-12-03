import pytest
import numpy as np
from sdforge.api.primitives import *
from sdforge.api.shaping import Round
from examples.primitives import *
from sdforge.api.render import SceneCompiler
from tests.conftest import requires_glsl_validator


def test_sphere_callable(headless_env):
    """Tests the numeric accuracy of the sphere's GPU evaluator."""
    s_callable = sphere(radius=1.0).to_callable()
    points = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]])
    expected = np.array([-1.0, -0.5, 0.0, 1.0])
    # Relax tolerance slightly for float32 GPU precision
    assert np.allclose(s_callable(points), expected, atol=1e-5)

def test_box_api():
    b1 = box()
    assert np.allclose(b1.size, (1,1,1))
    b2 = box(size=2.0)
    assert np.allclose(b2.size, (2,2,2))
    b3 = box(size=(1,2,3))
    assert np.allclose(b3.size, (1,2,3))

def test_box_callable(headless_env):
    b_callable = box(size=2.0).to_callable()
    points = np.array([[0, 0, 0], [1.5, 0, 0], [1, 1, 1], [2, 2, 0]])
    expected = np.array([-1.0, 0.5, 0.0, np.sqrt(1**2 + 1**2)])
    assert np.allclose(b_callable(points), expected, atol=1e-5)

def test_cylinder_callable(headless_env):
    c_callable = cylinder(radius=1.0, height=2.0).to_callable()
    points = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0], [1.5,0,0]])
    expected = np.array([-1.0, 0.0, 0.0, 0.0, 0.5])
    assert np.allclose(c_callable(points), expected, atol=1e-5)

def test_torus_api():
    t = torus(radius_major=2.0, radius_minor=0.5)
    assert t.radius_major == 2.0
    assert t.radius_minor == 0.5

def test_torus_callable():
    t_callable = torus(radius_major=1.0, radius_minor=0.2).to_callable()
    points = np.array([[1, 0, 0], [1, 0.2, 0], [1, 0.3, 0], [0, 0, 0]])
    expected = np.array([-0.2, 0.0, 0.1, 0.8])
    assert np.allclose(t_callable(points), expected)

def test_line_api():
    l1 = line(start=(0,0,0), end=(1,1,1), radius=0.2, rounded_caps=True)
    assert l1.rounded_caps is True
    l2 = line(start=(0,0,0), end=(1,1,1), radius=0.2, rounded_caps=False)
    assert l2.rounded_caps is False

def test_cylinder_callable():
    c_callable = cylinder(radius=1.0, height=2.0).to_callable()
    points = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0], [1.5,0,0]])
    expected = np.array([-1.0, 0.0, 0.0, 0.0, 0.5])
    assert np.allclose(c_callable(points), expected)

def test_plane_api():
    p = plane(normal=(0, 1, 0), point=(0, 1, 0))
    assert p.offset == -1.0

def test_plane_callable():
    p_callable = plane(normal=(0,1,0), point=(0,0.5,0)).to_callable()
    points = np.array([[10, 0.5, 0], [10, 1.5, 0], [10, -0.5, 0]])
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

def test_hex_prism_callable():
    h_callable = hex_prism(radius=1.0, height=0.5).to_callable()
    assert h_callable(np.array([[0,0,0]])) < 0
    assert h_callable(np.array([[0,0,0.3]])) > 0
    assert h_callable(np.array([[1.2,0,0]])) > 0

def test_pyramid_callable():
    p_callable = pyramid(height=1.0).to_callable()
    assert p_callable(np.array([[0,0,0]])) < 0
    assert p_callable(np.array([[0,0.6,0]])) > 0
    assert p_callable(np.array([[0,-0.6,0]])) > 0

def test_bezier_callable():
    c = curve(p0=(-1,0,0), p1=(0,1,0), p2=(1,0,0), radius=0.1)
    c_callable = c.to_callable()    
    points = np.array([
        [-1, 0, 0], # Start (surface dist = -radius)
        [1, 0, 0],  # End (surface dist = -radius)
        [0, 0.5, 0], # Midpoint roughly inside
        [0, 2, 0]   # Far away
    ])
    dists = c_callable(points)
    assert dists[0] < 0
    assert dists[1] < 0
    assert dists[2] < 0
    assert dists[3] > 0

def test_circle_callable_is_finite():
    """Tests that the circle is now a finite disc, not an infinite cylinder."""
    c_callable = circle(radius=1.0).to_callable()    
    assert c_callable(np.array([[0,0,0]])) < 0    
    z_dist = 10.0
    val = c_callable(np.array([[0,0,z_dist]]))
    assert np.isclose(val, z_dist - 0.001, atol=1e-4)
    assert val > 0

def test_rectangle_callable_is_finite():
    """Tests that the rectangle is now a finite plate, not an infinite prism."""
    r_callable = rectangle(size=(2.0, 2.0)).to_callable()
    
    # Point inside on Z=0
    assert r_callable(np.array([[0,0,0]])) < 0
    
    # Point "inside" the prism projection but far along Z
    z_dist = 10.0
    val = r_callable(np.array([[0,0,z_dist]]))
    assert np.isclose(val, z_dist - 0.001, atol=1e-4)
    assert val > 0

def test_triangle_callable_is_finite():
    """Tests that the triangle is a finite plate."""
    t_callable = triangle(radius=1.0).to_callable()
    assert t_callable(np.array([[0,0,0]])) < 0
    assert t_callable(np.array([[0,0,10]])) > 0

def test_trapezoid_callable_is_finite():
    tr_callable = trapezoid(bottom_width=1.0, top_width=0.5, height=0.5).to_callable()
    assert tr_callable(np.array([[0,0,0]])) < 0
    assert tr_callable(np.array([[0,0,10]])) > 0

def test_polyline_callable():
    """Tests polyline distance calculation."""
    points = [(0,0,0), (1,0,0), (1,1,0)]
    pl = polyline(points, radius=0.1)
    pl_callable = pl.to_callable()
    
    # Check points on the line segments
    test_points = np.array([
        [0,0,0], # Vertex 1
        [0.5,0,0], # Midpoint segment 1
        [1,0,0], # Vertex 2
        [1,0.5,0], # Midpoint segment 2
        [1,1,0] # Vertex 3
    ])
    
    dists = pl_callable(test_points)
    # All should be -radius (inside)
    assert np.allclose(dists, -0.1, atol=1e-5)
    
    # Check point far away
    far_point = np.array([[2,2,0]])
    # Distance to (1,1,0) is sqrt(2), minus radius
    expected_dist = np.sqrt(2) - 0.1
    assert np.allclose(pl_callable(far_point), expected_dist, atol=1e-5)

def test_polycurve_callable():
    """Tests polycurve distance calculation."""
    points = [(0,0,0), (1,1,0), (2,0,0)]
    pc = polycurve(points, radius=0.1, closed=False)
    pc_callable = pc.to_callable()
    assert np.isclose(pc_callable(np.array([[0,0,0]])), -0.1, atol=1e-5)    
    assert np.isclose(pc_callable(np.array([[2,0,0]])), -0.1, atol=1e-5)
    dist_corner = pc_callable(np.array([[1,1,0]]))
    assert dist_corner > -0.1

PRIMITIVE_TEST_CASES = [
    sphere(radius=1.2),
    box(size=(1.0, 0.5, 0.8)),
    torus(radius_major=1.0, radius_minor=0.25),
    line(start=(0,0,0), end=(0,1,0), radius=0.1, rounded_caps=True),
    line(start=(0,0,0), end=(0,1,0), radius=0.1, rounded_caps=False),
    cylinder(radius=0.5, height=1.5),
    cone(height=1.2, radius_base=0.6, radius_top=0.2),
    cone(height=1.2, radius_base=0.6, radius_top=0.0),
    plane(normal=(0.6, 0.8, 0), point=(0, 0.5, 0)),
    octahedron(size=1.3),
    ellipsoid(radii=(1.0, 0.5, 0.7)),
    circle(radius=1.5),
    rectangle(size=(1.0, 0.5)),
    triangle(radius=1.0),
    trapezoid(bottom_width=1.0, top_width=0.5, height=0.8),
    hex_prism(radius=1.0, height=0.5),
    pyramid(height=1.2),
    curve(p0=(0,0,0), p1=(0,1,0), p2=(1,0,0), radius=0.1),
    polyline([(0,0,0), (1,1,0), (2,0,0)], radius=0.1),
    polyline([(0,0,0), (1,0,0), (0,1,0)], radius=0.1, closed=True),
    polycurve([(0,0,0), (1,2,0), (2,0,0)], radius=0.1),
    polycurve([(0,0,0), (1,2,0), (2,0,0), (3,2,0)], radius=0.1, closed=True),
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


EXAMPLE_TEST_CASES = [
    (sphere_example, Sphere),
    (box_example, Round), # box().round() returns a Round node
    (cylinder_example, Cylinder),
    (torus_example, Torus),
    (cone_example, Cone),
    (hex_prism_example, HexPrism),
    (pyramid_example, Pyramid),
    (curve_example, Bezier),
    (circle_example, Circle),
    (rectangle_example, Rectangle),
    (triangle_example, Triangle),
    (trapezoid_example, Trapezoid),
    (polyline_example, Polyline),
    (polycurve_example, Polycurve),
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