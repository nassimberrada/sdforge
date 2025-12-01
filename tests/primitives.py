import pytest
import numpy as np
from sdforge import (
    SDFNode, sphere, box, torus, line, cylinder, cone, plane, 
    octahedron, ellipsoid, circle, rectangle,
    hex_prism, pyramid, curve, triangle, trapezoid, polyline, polycurve,
    Forge, Sketch
)
from sdforge.api.primitives import Primitive
from sdforge.api.operators import Operator
from sdforge.api.compositors import Compositor
from sdforge.api.scene import SceneCompiler
from tests.conftest import requires_glsl_validator, HEADLESS_SUPPORTED
from examples.primitives import (
    sphere_example, box_example, cylinder_example, torus_example, 
    cone_example, hex_prism_example, pyramid_example, curve_example, 
    circle_example, rectangle_example, triangle_example, 
    trapezoid_example, polyline_example, polycurve_example
)

def test_sphere_api():
    s1 = sphere()
    assert isinstance(s1, Primitive)
    assert s1.name == "Sphere"
    assert s1.radius == 1.0
    assert s1.params[0] == 1.0
    s2 = sphere(radius=1.5)
    assert s2.radius == 1.5

def test_sphere_callable():
    s_callable = sphere(radius=1.0).to_callable()
    points = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]])
    expected = np.array([-1.0, -0.5, 0.0, 1.0])
    assert np.allclose(s_callable(points), expected)

def test_box_api():
    b1 = box(); assert np.allclose(b1.size, (1,1,1))
    b2 = box(size=2.0); assert np.allclose(b2.size, (2,2,2))
    b3 = box(size=(1,2,3)); assert np.allclose(b3.size, (1,2,3))

def test_box_callable():
    b_callable = box(size=2.0).to_callable()
    points = np.array([[0, 0, 0], [1.5, 0, 0], [1, 1, 1], [2, 2, 0]])
    expected = np.array([-1.0, 0.5, 0.0, np.sqrt(1**2 + 1**2)])
    assert np.allclose(b_callable(points), expected)

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
    points = np.array([[-1, 0, 0], [1, 0, 0], [0, 0.5, 0], [0, 2, 0]])
    dists = c_callable(points)
    assert dists[0] < 0
    assert dists[1] < 0
    assert dists[2] < 0
    assert dists[3] > 0

def test_circle_callable_is_finite():
    c_callable = circle(radius=1.0).to_callable()
    assert c_callable(np.array([[0,0,0]])) < 0
    z_dist = 10.0
    val = c_callable(np.array([[0,0,z_dist]]))
    assert np.isclose(val, z_dist - 0.001, atol=1e-4)
    assert val > 0

def test_rectangle_callable_is_finite():
    r_callable = rectangle(size=(2.0, 2.0)).to_callable()    
    assert r_callable(np.array([[0,0,0]])) < 0    
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
    pl = polyline([(0,0,0), (1,0,0), (1,1,0)], radius=0.1)
    assert isinstance(pl, Compositor)
    pl_callable = pl.to_callable()
    test_points = np.array([[0,0,0], [0.5,0,0], [1,0,0], [1,0.5,0], [1,1,0]])
    assert np.allclose(pl_callable(test_points), -0.1, atol=1e-5)
    assert np.allclose(pl_callable(np.array([[2,2,0]])), np.sqrt(2) - 0.1, atol=1e-5)

def test_polycurve_callable():
    pc = polycurve([(0,0,0), (1,1,0), (2,0,0)], radius=0.1, closed=False)
    # Consolidated Polycurve returns Compositor
    assert isinstance(pc, Compositor)
    pc_callable = pc.to_callable()
    assert np.isclose(pc_callable(np.array([[0,0,0]])), -0.1, atol=1e-5)
    assert np.isclose(pc_callable(np.array([[2,0,0]])), -0.1, atol=1e-5)
    assert pc_callable(np.array([[1,1,0]])) > -0.1

def test_forge_api():
    f = Forge("length(p) - 1.0")
    scene_code = SceneCompiler().compile(f)
    assert f.unique_id in scene_code
    assert "length(p) - 1.0" in scene_code
    assert f"vec4 var_0 = vec4({f.unique_id}(p), -1.0, 0.0, 0.0);" in scene_code

def test_forge_with_uniforms_api():
    uniforms = {'u_radius': 1.5, 'u_offset': 0.1}
    f = Forge("length(p) - u_radius + u_offset", uniforms=uniforms)
    scene_code = SceneCompiler().compile(f)
    assert f"in float u_radius, in float u_offset" in scene_code
    assert f"vec4 var_0 = vec4({f.unique_id}(p, u_radius, u_offset), -1.0, 0.0, 0.0);" in scene_code

def test_forge_callable_fails_with_uniforms():
    f = Forge("length(p) - u_radius", uniforms={'u_radius': 1.0})
    with pytest.raises(TypeError, match="Cannot create a callable for a Forge object with uniforms"):
        f.to_callable()

@pytest.mark.skipif(not HEADLESS_SUPPORTED, reason="Requires moderngl/glfw.")
def test_forge_callable_equivalence():
    try:
        import glfw
        if not glfw.init(): pytest.skip("glfw.init() failed")
    except Exception: pytest.skip("glfw issue")
    radius = 1.2
    forge_sphere = Forge(f"length(p) - {radius}")
    native_sphere = sphere(radius)
    try: forge_callable = forge_sphere.to_callable()
    except RuntimeError: pytest.skip("Context fail")
    native_callable = native_sphere.to_callable()
    points = (np.random.rand(1024, 3) * 4 - 2).astype('f4')
    assert np.allclose(forge_callable(points), native_callable(points), atol=1e-5)

def test_sketch_init_and_chaining():
    s = Sketch(start=(1, 2))
    assert np.allclose(s._current_pos, [1, 2])
    res = s.line_to(1, 0).curve_to(2, 1, (1.5, 0)).close()
    assert res is s
    assert len(s._segments) == 3

def test_sketch_segments_data():
    s = Sketch(start=(0, 0)).line_to(1, 0).curve_to(1, 1, control=(2, 0))
    seg0 = s._segments[0]
    assert seg0['type'] == 'line'
    assert np.allclose(seg0['end'], [1, 0, 0])
    seg1 = s._segments[1]
    assert seg1['type'] == 'bezier'
    assert np.allclose(seg1['control'], [2, 0, 0])

def test_sketch_to_sdf_and_callable():
    s = Sketch(start=(0,0)).line_to(2,0).to_sdf(stroke_radius=0.1)
    assert isinstance(s, Compositor)
    callable_fn = s.to_callable()
    points = np.array([[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [3.0, 0.0, 0.0]])
    dists = callable_fn(points)
    assert np.isclose(dists[0], -0.1, atol=1e-5)
    assert np.isclose(dists[1], 0.1, atol=1e-5)
    assert np.isclose(dists[2], 0.9, atol=1e-5)

def test_sketch_profile_logic():
    sketch = Sketch().line_to(1,0).to_sdf(stroke_radius=0.1)
    prof_callable = sketch.to_profile_callable()
    p = np.array([[0.5, 0.0, 100.0]]) 
    d = prof_callable(p)
    assert np.isclose(d[0], -0.1, atol=1e-5)

PRIMITIVE_TEST_CASES = [
    sphere(1.2), box((1.0, 0.5, 0.8)), torus(1.0, 0.25),
    line((0,0,0), (0,1,0), 0.1, True), cylinder(0.5, 1.5),
    cone(1.2, 0.6, 0.2), plane((0.6, 0.8, 0), (0, 0.5, 0)),
    octahedron(1.3), ellipsoid((1.0, 0.5, 0.7)),
    circle(1.5), rectangle((1.0, 0.5)), triangle(1.0),
    trapezoid(1.0, 0.5, 0.8), hex_prism(1.0, 0.5), pyramid(1.2),
    curve((0,0,0), (0,1,0), (1,0,0), 0.1),
    polyline([(0,0,0), (1,1,0), (2,0,0)], 0.1),
    polycurve([(0,0,0), (1,2,0), (2,0,0)], 0.1),
    Forge("length(p) - 1.0"),
    Sketch().line_to(1,0).to_sdf()
]

@pytest.mark.usefixtures("assert_equivalence")
@pytest.mark.parametrize("sdf_obj", PRIMITIVE_TEST_CASES, ids=[type(p).__name__ for p in PRIMITIVE_TEST_CASES])
def test_primitive_equivalence(assert_equivalence, sdf_obj):
    assert_equivalence(sdf_obj)

@requires_glsl_validator
@pytest.mark.parametrize("sdf_obj", PRIMITIVE_TEST_CASES, ids=[type(p).__name__ for p in PRIMITIVE_TEST_CASES])
def test_primitive_glsl_compiles(validate_glsl, sdf_obj):
    scene_code = SceneCompiler().compile(sdf_obj)
    validate_glsl(scene_code, sdf_obj)

EXAMPLE_TEST_CASES = [
    (sphere_example, Primitive), (box_example, Operator),
    (cylinder_example, Primitive),
    (torus_example, Primitive), (cone_example, Primitive), (hex_prism_example, Primitive),
    (pyramid_example, Primitive), (curve_example, Primitive), (circle_example, Primitive),
    (rectangle_example, Primitive), (triangle_example, Primitive), (trapezoid_example, Primitive),
    (polyline_example, Compositor), (polycurve_example, Compositor),
]

@pytest.mark.parametrize("example_func, expected_class", EXAMPLE_TEST_CASES, ids=[f[0].__name__ for f in EXAMPLE_TEST_CASES])
def test_primitive_example_runs(example_func, expected_class):
    scene = example_func()
    assert isinstance(scene, SDFNode)
    assert isinstance(scene, expected_class)