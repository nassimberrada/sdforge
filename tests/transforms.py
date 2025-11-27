import pytest
import numpy as np
from sdforge import sphere, box, X, Y, Z
from sdforge.api.transforms import Warp
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

def test_masked_translate_callable(shape):
    """Tests translation with a mask."""
    offset = np.array([10, 0, 0])
    # Mask is a sphere at origin radius 1.
    mask = sphere(1.0) 
    t_shape = shape.translate(offset, mask=mask, mask_falloff=0.0)
    t_callable = t_shape.to_callable()
    
    # Point inside mask (0,0,0) -> should be translated (-10) before child eval
    # box(1,2,3) at origin.
    # p = (0,0,0). mask(p) = -1. factor=1. p_trans = p-10 = (-10,0,0).
    # box dist at (-10,0,0) should be approx 10 - 0.5 = 9.5
    p_in = np.array([[0,0,0]])
    d_in = t_callable(p_in)
    assert d_in > 5.0 
    
    # Point outside mask (5,0,0) -> mask(p) = 4. factor=0. p_trans = p.
    # box dist at (5,0,0) approx 4.5.
    p_out = np.array([[5,0,0]])
    d_out = t_callable(p_out)
    assert np.isclose(d_out, 4.5, atol=0.1)

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

def test_masked_scale_callable(shape):
    """Tests scaling with a mask."""
    factor = 0.1 # Shrink
    mask = sphere(1.0)
    s_shape = shape.scale(factor, mask=mask)
    s_callable = s_shape.to_callable()
    
    # Inside mask: scaled down.
    # box is 1x2x3. Scaled by 0.1 becomes 0.1x0.2x0.3.
    # p=(0.2,0,0) is outside the tiny box.
    p_in = np.array([[0.2, 0, 0]]) 
    d_in = s_callable(p_in)
    # distance should be (0.2 - 0.05) * 0.1? No, dist = d_sdf * scale_corr
    # p/f = 2.0. Box edge 0.5. Dist=1.5. * 0.1 = 0.15.
    assert np.isclose(d_in, 0.15, atol=0.01)
    
    # Outside mask: normal size.
    # p=(5,0,0). Normal box edge 0.5. Dist 4.5.
    p_out = np.array([[5, 0, 0]])
    d_out = s_callable(p_out)
    assert np.isclose(d_out, 4.5, atol=0.1)

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

def test_rotate_arbitrary_axis(shape):
    """Tests rotation around a non-cardinal axis."""
    angle = np.pi / 4
    axis = np.array([1.0, 1.0, 0.0])
    r_shape = shape.rotate(axis, angle)

    r_callable = r_shape.to_callable()
    point = np.array([[2.0, 1.0, 1.0]])

    # Construct matrix for test verification (Rodrigues)
    axis = axis / np.sqrt(2)
    kx, ky, kz = axis
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3) + s * K + (1 - c) * (K @ K)

    expected = shape.to_callable()(point @ R.T)
    assert np.allclose(r_callable(point), expected)

def test_orient_callable(shape):
    o_shape = shape.orient(X)
    o_callable = o_shape.to_callable()
    point = np.array([[3.1, 2.1, 1.1]])
    expected = shape.to_callable()(point[:, [2, 1, 0]])
    assert np.allclose(o_callable(point), expected)

def test_twist_callable(shape):
    k = 5.0
    t_shape = shape.twist(strength=k)
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

def test_masked_twist_callable(shape):
    """Tests twist with a mask."""
    k = 5.0
    # Mask is box at y=10.
    mask = box(1.0).translate((0, 10, 0))
    t_shape = shape.twist(strength=k, mask=mask)
    t_callable = t_shape.to_callable()
    
    # Point at origin (far from mask) -> Untwisted
    p_orig = np.array([[0.5, 0, 0]]) # edge of box
    d_orig = t_callable(p_orig)
    assert np.isclose(d_orig, 0.0, atol=1e-4)
    
    # Point at y=10 (inside mask) -> Twisted
    # At y=10, angle = 5 * 10 = 50. 
    # If we probe at a point that IS on the twisted surface... hard to calculate analytically.
    # Instead, we verify it is DIFFERENT from untwisted.
    p_mask = np.array([[0.5, 10, 0]])
    d_mask = t_callable(p_mask)
    # The original box is straight up. The twisted box surface has rotated.
    # So distance to original surface point should be non-zero.
    assert abs(d_mask) > 0.01

def test_bend_callable(shape):
    k = 0.5
    b_shape = shape.bend(Y, curvature=k)
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

# --- Warp Tests ---

def test_warp_api_structure():
    """Tests the fluent API and object structure."""
    s = sphere(1.0)
    warped_s = s.warp(frequency=2.0, strength=0.5)

    assert isinstance(warped_s, Warp)
    assert warped_s.child == s
    assert warped_s.frequency == 2.0
    assert warped_s.strength == 0.5

def test_warp_glsl_generation():
    """Tests that the GLSL string contains the correct function calls."""
    s = sphere(1.0).warp(frequency=3.0, strength=1.0)
    scene_code = SceneCompiler().compile(s)

    assert "opWarp" in scene_code
    # Check that dependencies are met (snoiseVec3 from noise.glsl)
    assert "snoiseVec3" in scene_code 

def test_warp_callable_raises_error():
    """
    Ensures that calling .to_callable() (used for meshing) raises a TypeError
    because procedural noise is not yet supported in NumPy.
    """
    s = sphere(1.0).warp(frequency=2.0, strength=0.5)
    with pytest.raises(TypeError, match="Cannot create a callable"):
        s.to_callable()

@requires_glsl_validator
def test_warp_glsl_compiles(validate_glsl):
    """
    Validates that the generated GLSL for warping is syntactically correct
    and that all dependencies (noise functions) are included.
    """
    s = box(1.0).warp(frequency=1.5, strength=0.2)
    scene_code = SceneCompiler().compile(s)
    validate_glsl(scene_code)


# --- Equivalence and Compilation Tests ---

TRANSFORM_TEST_CASES = [
    sphere(radius=0.8).translate((0.5, -0.2, 0.1)),
    box(size=1.0).scale(2.0),
    box(size=1.0).scale((0.5, 1.0, 1.5)),
    sphere(radius=0.8).rotate(X, np.pi / 4),
    sphere(radius=0.8).rotate(Y, np.pi / 2),
    sphere(radius=0.8).rotate(Z, np.pi),
    sphere(radius=0.8).rotate((1,1,0), np.pi / 4), # Arbitrary axis
    sphere(radius=0.8).orient(X),
    sphere(radius=0.8).twist(strength=5.0),
    box(size=1.0).bend(Y, curvature=0.5),
    sphere(radius=0.5).repeat((2, 2, 0)),
    sphere(radius=0.5).limited_repeat((1.5, 0, 0), (2, 0, 0)),
    box(size=0.2).translate((1,0,0)).polar_repeat(8),
    box(size=0.4).mirror(X | Y),
    # Test chaining
    box(size=1.0).translate((1,0,0)).scale(2.0).rotate(Y, 1.0),
    # Masked transforms
    box(1.0).translate((1,0,0), mask=sphere(0.5)),
    box(1.0).scale(2.0, mask=sphere(0.5)),
    box(1.0).twist(2.0, mask=sphere(0.5), mask_falloff=0.1),
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