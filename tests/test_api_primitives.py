import pytest
from sdforge import *
import numpy as np
from unittest.mock import MagicMock

# --- API Usage Tests  ---

def test_sphere():
    s = sphere()
    assert isinstance(s, SDFObject)
    s_rad = sphere(1.5)
    assert s_rad.r == 1.5

def test_box():
    b = box()
    assert isinstance(b, SDFObject)
    b_size = box(size=(1, 2, 3))
    assert b_size.size == (1, 2, 3)
    b_uniform = box(size=2)
    assert b_uniform.size == (2, 2, 2)

def test_rounded_box():
    rb = rounded_box(size=(1, 2, 3), radius=0.2)
    assert isinstance(rb, SDFObject)
    assert rb.size == (1, 2, 3)
    assert rb.radius == 0.2

def test_torus():
    t = torus(major=2.0, minor=0.5)
    assert isinstance(t, SDFObject)
    assert t.major == 2.0
    assert t.minor == 0.5

def test_capsule():
    c = capsule(a=(0, 0, 0), b=(0, 1, 0), radius=0.1)
    assert isinstance(c, SDFObject)
    assert np.array_equal(c.a, np.array([0, 0, 0]))
    assert np.array_equal(c.b, np.array([0, 1, 0]))
    assert c.radius == 0.1

def test_cylinder():
    c = cylinder(radius=0.5, height=2.0)
    assert isinstance(c, SDFObject)
    assert c.radius == 0.5
    assert c.height == 2.0

def test_cone():
    c = cone(height=1.5, radius=0.7)
    assert isinstance(c, SDFObject)
    assert c.height == 1.5
    assert c.radius == 0.7

def test_plane():
    p = plane(normal=X, offset=1.0)
    assert isinstance(p, SDFObject)
    assert np.array_equal(p.normal, X)
    assert p.offset == 1.0

def test_hex_prism():
    hp = hex_prism(radius=0.8, height=0.3)
    assert isinstance(hp, SDFObject)
    assert hp.radius == 0.8
    assert hp.height == 0.3

def test_octahedron():
    o = octahedron(size=1.2)
    assert isinstance(o, SDFObject)
    assert o.size == 1.2

def test_ellipsoid():
    e = ellipsoid(radii=(1, 2, 3))
    assert isinstance(e, SDFObject)
    assert e.radii == (1, 2, 3)

def test_box_frame():
    bf = box_frame(size=(1, 1, 2), edge_radius=0.05)
    assert isinstance(bf, SDFObject)
    assert bf.size == (1, 1, 2)
    assert bf.edge_radius == 0.05

def test_capped_torus():
    ct = capped_torus(angle_sc=(0.8, 0.6), major_radius=1.5, minor_radius=0.2)
    assert isinstance(ct, SDFObject)
    assert np.array_equal(ct.angle_sc, np.array([0.8, 0.6]))
    assert ct.major_radius == 1.5

def test_link():
    l = link(length=1.0, radius1=0.4, radius2=0.1)
    assert isinstance(l, SDFObject)
    assert l.length == 1.0
    assert l.radius1 == 0.4

def test_capped_cylinder():
    cc = capped_cylinder(a=X, b=Y, radius=0.2)
    assert isinstance(cc, SDFObject)
    assert np.array_equal(cc.a, X)

def test_rounded_cylinder():
    rc = rounded_cylinder(radius=0.5, round_radius=0.1, height=1.5)
    assert isinstance(rc, SDFObject)
    assert rc.radius == 0.5

def test_capped_cone():
    cc = capped_cone(height=2.0, radius1=0.5, radius2=0.2)
    assert isinstance(cc, SDFObject)
    assert cc.radius2 == 0.2

def test_round_cone():
    rc = round_cone(radius1=0.6, radius2=0.1, height=1.0)
    assert isinstance(rc, SDFObject)
    assert rc.height == 1.0

def test_pyramid():
    p = pyramid(height=1.5)
    assert isinstance(p, SDFObject)
    assert p.height == 1.5

def test_forge():
    f = Forge("return length(p) - 1.0;")
    assert isinstance(f, SDFObject)
    assert "length(p) - 1.0" in f.glsl_code_body

# --- Numeric Accuracy Tests  ---

def test_sphere_callable():
    s_callable = sphere(r=1.0).to_callable()
    points = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]])
    expected = np.array([-1.0, -0.5, 0.0, 1.0])
    assert np.allclose(s_callable(points), expected)

def test_box_callable():
    b_callable = box(size=2.0).to_callable()
    points = np.array([[0, 0, 0], [1.5, 0, 0], [1, 1, 1], [2, 2, 0]])
    expected = np.array([-1.0, 0.5, 0.0, np.sqrt(1**2 + 1**2)])
    assert np.allclose(b_callable(points), expected)

def test_rounded_box_callable():
    rb_callable = rounded_box(size=2.0, radius=0.2).to_callable()
    points = np.array([[0, 0, 0], [1.2, 0.5, 0]])
    expected = np.array([-0.2, 0.0])
    assert np.allclose(rb_callable(points), expected)

def test_torus_callable():
    t_callable = torus(major=1.0, minor=0.2).to_callable()
    points = np.array([
        [1, 0, 0],
        [1, 0.2, 0],
        [1, 0.3, 0],
        [0, 0, 0]
    ])
    expected = np.array([-0.2, 0.0, 0.1, 0.8])
    assert np.allclose(t_callable(points), expected)

def test_capsule_callable():
    c_callable = capsule(a=[0, -1, 0], b=[0, 1, 0], radius=0.5).to_callable()
    points = np.array([
        [0, 0, 0],
        [0.5, 0, 0],
        [0, 1.5, 0],
        [0, 2, 0]
    ])
    expected = np.array([-0.5, 0.0, 0.0, 0.5])
    assert np.allclose(c_callable(points), expected)

def test_cylinder_callable():
    c_callable = cylinder(radius=1.0, height=2.0).to_callable()
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1.5, 0, 0]
    ])
    expected = np.array([-1.0, 0.0, 0.0, 0.0, 0.5])
    assert np.allclose(c_callable(points), expected)

def test_plane_callable():
    p_callable = plane(normal=Y, offset=0.5).to_callable()
    points = np.array([
        [10, -0.5, 0],
        [10, 0.5, 0],
        [10, -1.5, 0]
    ])
    expected = np.array([0.0, 1.0, -1.0])
    assert np.allclose(p_callable(points), expected)

def test_octahedron_callable():
    o_callable = octahedron(size=1.0).to_callable()
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0.5, 0]
    ])
    factor = 0.57735027
    expected = np.array([-1.0 * factor, 0.0, 0.0])
    assert np.allclose(o_callable(points), expected)

def test_ellipsoid_callable():
    e_callable = ellipsoid(radii=(1, 2, 3)).to_callable()
    points = np.array([
        [0.1, 0.1, 0.1],
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ])
    actual = e_callable(points)
    assert actual[0] < 0
    assert np.allclose(actual[1:], [0.0, 0.0, 0.0], atol=1e-6)

# --- Equivalence Tests  ---

np.random.seed(42)
test_points = np.random.rand(1000, 3) * 4 - 2  # Points in a [-2, 2] cube

PRIMITIVE_TEST_CASES = [
    (sphere(1.2), "sdSphere(p, 1.2)"),
    (box(1.5), "sdBox(p, vec3(0.75))"),
    (box((1, 2, 3)), "sdBox(p, vec3(0.5, 1.0, 1.5))"),
    (torus(1.0, 0.25), "sdTorus(p, vec2(1.0, 0.25))"),
    (octahedron(1.3), "sdOctahedron(p, 1.3)"),
    (cylinder(0.5, 2.0), "sdCylinder(p, vec2(0.5, 1.0))"),
]

try:
    import moderngl
    import glfw
    _HEADLESS_RENDERING_SUPPORTED = True
except ImportError:
    _HEADLESS_RENDERING_SUPPORTED = False

requires_headless = pytest.mark.skipif(
    not _HEADLESS_RENDERING_SUPPORTED,
    reason="Equivalence tests require moderngl and glfw for headless GLSL evaluation."
)

@requires_headless
@pytest.mark.parametrize("sdf_obj, glsl_expr", PRIMITIVE_TEST_CASES, ids=[type(c[0]).__name__ for c in PRIMITIVE_TEST_CASES])
def test_primitive_equivalence(sdf_obj, glsl_expr, monkeypatch):
    """
    Tests that a primitive's to_callable() (NumPy) matches its GLSL equivalent.
    """
    monkeypatch.delattr('sdforge.api.Forge._mgl_context', raising=False)
    monkeypatch.setattr('sdforge.api._MODERNGL_AVAILABLE', True)

    mock_glfw = MagicMock()
    mock_mgl = MagicMock()
    
    mock_context = MagicMock()
    mock_mgl.create_context.return_value = mock_context

    monkeypatch.setattr('sdforge.api.glfw', mock_glfw)
    monkeypatch.setattr('sdforge.api.moderngl', mock_mgl)

    numpy_distances = sdf_obj.to_callable()(test_points)

    from sdforge.api import _get_glsl_content
    primitives_glsl = _get_glsl_content("primitives.glsl")
    
    glsl_shape = Forge(f"""
        {primitives_glsl}
        return {glsl_expr};
    """)
    glsl_callable = glsl_shape.to_callable()
    
    assert callable(glsl_callable)
    assert callable(sdf_obj.to_callable())