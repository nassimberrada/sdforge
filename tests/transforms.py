import pytest
import numpy as np
from sdforge import sphere, box, X, Y, Z
from sdforge.render import SceneCompiler
from tests.conftest import requires_glsl_validator

@pytest.fixture
def shape():
    """Provides a basic, non-symmetrical shape for transform tests."""
    return box(size=(1, 2, 3))

# --- API and Callable Tests ---

def test_translate_api_and_callable(shape):
    offset = np.array([1, 2, 3])
    t_shape = shape.translate(offset)
    t_op = shape + offset

    t_callable = t_shape.to_callable()
    point = np.array([[1.1, 2.2, 3.3]])
    expected = shape.to_callable()(point - offset)
    assert np.allclose(t_callable(point), expected)
    assert np.allclose(t_op.to_callable()(point), expected)

def test_scale_api_and_callable(shape):
    factor = 2.0
    s_shape = shape.scale(factor)
    s_op1 = shape * factor
    s_op2 = factor * shape
    
    s_callable = s_shape.to_callable()
    point = np.array([[0.6, 1.2, 1.8]])
    expected = shape.to_callable()(point / factor) * factor
    assert np.allclose(s_callable(point), expected)
    assert np.allclose(s_op1.to_callable()(point), expected)
    assert np.allclose(s_op2.to_callable()(point), expected)

def test_rotate_api_and_callable(shape):
    angle = np.pi / 2
    r_shape = shape.rotate(Z, angle)
    
    r_callable = r_shape.to_callable()
    point = np.array([[2.5, 0.6, 0]])
    c, s = np.cos(angle), np.sin(angle)
    # Inverse rotation matrix
    rot_matrix = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    expected = shape.to_callable()(point @ rot_matrix.T)
    assert np.allclose(r_callable(point), expected)

def test_orient_callable(shape):
    o_shape = shape.orient(X)
    o_callable = o_shape.to_callable()
    point = np.array([[3.1, 2.1, 1.1]])
    expected = shape.to_callable()(point[:, [2, 1, 0]])
    assert np.allclose(o_callable(point), expected)

def test_twist_callable(shape):
    k = 5.0
    t_shape = shape.twist(k)
    t_callable = t_shape.to_callable()
    point = np.array([[0.1, 0.2, 0.3]])
    p = point
    # Note: twist is its own inverse if you flip the angle, but the Python logic
    # applies the forward transform for simplicity here. This test is primarily
    # for shape and correctness, while the equivalence test confirms it matches GLSL.
    c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
    x_new, z_new = p[:,0]*c - p[:,2]*s, p[:,0]*s + p[:,2]*c
    q = np.stack([x_new, p[:,1], z_new], axis=-1)
    expected = shape.to_callable()(q)
    assert np.allclose(t_callable(point), expected)

def test_bend_callable(shape):
    k = 0.5
    b_shape = shape.bend(Y, k)
    b_callable = b_shape.to_callable()
    point = np.array([[0.1, 0.2, 0.3]])
    p = point
    c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
    # Apply the INVERSE bend transformation for the test assertion
    x_new, z_new = c * p[:,0] - s * p[:,2], s * p[:,0] + c * p[:,2]
    q = np.stack([x_new, p[:,1], z_new], axis=-1)
    expected = shape.to_callable()(q)
    assert np.allclose(b_callable(point), expected)

def test_mirror_callable(shape):
    m_shape = shape.mirror(X | Z)
    m_callable = m_shape.to_callable()
    point = np.array([[-0.1, 0.2, -0.3]])
    expected = shape.to_callable()(np.abs(point))
    assert np.allclose(m_callable(point), expected)


# --- Equivalence and Compilation Tests ---

TRANSFORM_TEST_CASES = [
    sphere(0.8).translate((0.5, -0.2, 0.1)),
    box(1.0).scale(2.0),
    box(1.0).scale((0.5, 1.0, 1.5)),
    sphere(0.8).rotate(X, np.pi / 4),
    sphere(0.8).rotate(Y, np.pi / 2),
    sphere(0.8).rotate(Z, np.pi),
    sphere(0.8).orient(X),
    sphere(0.8).twist(5.0),
    box(1.0).bend(Y, 0.5),
    sphere(0.5).repeat((2, 2, 0)),
    sphere(0.5).limited_repeat((1.5, 0, 0), (2, 0, 0)),
    box(0.2).translate((1,0,0)).polar_repeat(8),
    box(0.4).mirror(X | Y),
    # Test chaining
    box(1.0).translate((1,0,0)).scale(2.0).rotate(Y, 1.0)
]

@pytest.mark.usefixtures("assert_equivalence")
@pytest.mark.parametrize("sdf_obj", TRANSFORM_TEST_CASES, ids=[repr(o) for o in TRANSFORM_TEST_CASES])
def test_transform_equivalence(assert_equivalence, sdf_obj):
    """Tests numeric equivalence between Python and GLSL for all transforms."""
    assert_equivalence(sdf_obj)

@requires_glsl_validator
@pytest.mark.parametrize("sdf_obj", TRANSFORM_TEST_CASES, ids=[repr(o) for o in TRANSFORM_TEST_CASES])
def test_transform_glsl_compiles(validate_glsl, sdf_obj):
    """Tests that the GLSL generated for all transforms is syntactically valid."""
    scene_code = SceneCompiler().compile(sdf_obj)
    validate_glsl(scene_code)