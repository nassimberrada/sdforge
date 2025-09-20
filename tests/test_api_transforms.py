import pytest
from sdforge import *
import numpy as np

@pytest.fixture
def shape():
    return box(size=(1, 2, 3))

def test_translate(shape):
    t = shape.translate((1, 2, 3))
    assert isinstance(t, SDFObject)
    assert "opTranslate(" in t.to_glsl()

def test_scale(shape):
    s = shape.scale(2.0)
    assert isinstance(s, SDFObject)
    assert "opScale(" in s.to_glsl()

def test_orient(shape):
    o_x = shape.orient(X)
    assert isinstance(o_x, SDFObject)
    assert "p.zyx" in o_x.to_glsl()

    o_y = shape.orient(Y)
    assert isinstance(o_y, SDFObject)
    assert "p.xzy" in o_y.to_glsl()

    o_z = shape.orient(Z)
    assert isinstance(o_z, SDFObject)
    assert o_z.to_glsl() == shape.to_glsl()


def test_rotate(shape):
    r = shape.rotate(Y, np.pi / 2)
    assert isinstance(r, SDFObject)
    assert "opRotateY(" in r.to_glsl()

def test_twist(shape):
    t = shape.twist(5.0)
    assert isinstance(t, SDFObject)
    assert "opTwist(" in t.to_glsl()

def test_shear(shape):
    sh_xy = shape.shear_xy((0.1, 0.2))
    sh_xz = shape.shear_xz((0.3, 0.4))
    sh_yz = shape.shear_yz((0.5, 0.6))
    assert isinstance(sh_xy, SDFObject) and "opShearXY" in sh_xy.to_glsl()
    assert isinstance(sh_xz, SDFObject) and "opShearXZ" in sh_xz.to_glsl()
    assert isinstance(sh_yz, SDFObject) and "opShearYZ" in sh_yz.to_glsl()

def test_bend(shape):
    b_x = shape.bend_x(0.5)
    b_y = shape.bend_y(0.5)
    b_z = shape.bend_z(0.5)
    assert isinstance(b_x, SDFObject) and "opBendX" in b_x.to_glsl()
    assert isinstance(b_y, SDFObject) and "opBendY" in b_y.to_glsl()
    assert isinstance(b_z, SDFObject) and "opBendZ" in b_z.to_glsl()

def test_repeat(shape):
    r = shape.repeat((2, 0, 0))
    assert isinstance(r, SDFObject)
    assert "opRepeat(" in r.to_glsl()

def test_limited_repeat(shape):
    lr = shape.limited_repeat(spacing=(1.2, 0, 0), limits=(3, 0, 0))
    assert isinstance(lr, SDFObject)
    assert "opLimitedRepeat(" in lr.to_glsl()

def test_polar_repeat(shape):
    pr = shape.polar_repeat(6)
    assert isinstance(pr, SDFObject)
    assert "opPolarRepeat(" in pr.to_glsl()

def test_mirror(shape):
    m = shape.mirror(X)
    assert isinstance(m, SDFObject)
    assert "opMirror(" in m.to_glsl()

def test_elongate(shape):
    e = shape.elongate((0.1, 0.2, 0.3))
    assert isinstance(e, SDFObject)
    assert "opElongate(" in e.to_glsl()

def test_translate_callable(shape):
    offset = np.array([1, 2, 3])
    t_shape = shape.translate(offset)
    t_callable = t_shape.to_callable()
    point = np.array([[1.1, 2.2, 3.3]])
    expected = shape.to_callable()(point - offset)
    assert np.allclose(t_callable(point), expected)

def test_scale_callable(shape):
    factor = 2.0
    s_shape = shape.scale(factor)
    s_callable = s_shape.to_callable()
    point = np.array([[0.6, 1.2, 1.8]])
    expected = shape.to_callable()(point / factor) * factor
    assert np.allclose(s_callable(point), expected)

def test_rotate_callable(shape):
    angle = np.pi / 2
    r_shape = shape.rotate(Z, angle)
    r_callable = r_shape.to_callable()
    point = np.array([[2.5, 0.6, 0]])
    c, s = np.cos(angle), np.sin(angle)
    # Inverse rotation matrix
    rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    expected = shape.to_callable()(point @ rot_matrix.T)
    assert np.allclose(r_callable(point), expected)

def test_orient_callable(shape):
    o_shape = shape.orient(X)
    o_callable = o_shape.to_callable()
    point = np.array([[3.1, 2.1, 1.1]])
    expected = shape.to_callable()(point[:, [2, 1, 0]])
    assert np.allclose(o_callable(point), expected)

def test_mirror_callable(shape):
    m_shape = shape.mirror(X | Z)
    m_callable = m_shape.to_callable()
    point = np.array([[-0.1, 0.2, -0.3]])
    expected = shape.to_callable()(np.abs(point))
    assert np.allclose(m_callable(point), expected)

def test_twist_callable(shape):
    k = 5.0
    t_shape = shape.twist(k)
    t_callable = t_shape.to_callable()
    point = np.array([[0.1, 0.2, 0.3]])
    p = point
    c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
    x_new = p[:,0]*c - p[:,2]*s
    z_new = p[:,0]*s + p[:,2]*c
    q = np.stack([x_new, p[:,1], z_new], axis=-1)
    expected = shape.to_callable()(q)
    assert np.allclose(t_callable(point), expected)

def test_bend_callable(shape):
    k = 0.5
    b_shape = shape.bend_y(k)
    b_callable = b_shape.to_callable()
    point = np.array([[0.1, 0.2, 0.3]])
    p = point
    c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
    x_new = c * p[:,0] + s * p[:,2]
    z_new = -s * p[:,0] + c * p[:,2]
    q = np.stack([x_new, p[:,1], z_new], axis=-1)
    expected = shape.to_callable()(q)
    assert np.allclose(b_callable(point), expected)

def test_repeat_callable(shape):
    spacing = np.array([2, 0, 0])
    r_shape = shape.repeat(spacing)
    r_callable = r_shape.to_callable()
    point = np.array([[2.1, 0.2, 0.3]])
    q = point.copy()
    mask = spacing != 0
    p_masked = point[:, mask]
    s_masked = spacing[mask]
    q[:, mask] = np.mod(p_masked + 0.5 * s_masked, s_masked) - 0.5 * s_masked    
    expected = shape.to_callable()(q)
    assert np.allclose(r_callable(point), expected)

def test_limited_repeat_callable(shape):
    spacing = np.array([1.2, 0, 0])
    limits = np.array([3, 0, 0])
    lr_shape = shape.limited_repeat(spacing=spacing, limits=limits)
    lr_callable = lr_shape.to_callable()
    point = np.array([[2.5, 0.2, 0.3]])
    
    q = point.copy()
    mask = spacing != 0
    if np.any(mask):
        p_masked = point[:, mask]
        s_masked = spacing[mask]
        l_masked = limits[mask]
        q[:, mask] = p_masked - s_masked * np.clip(np.round(p_masked / s_masked), -l_masked, l_masked)
        
    expected = shape.to_callable()(q)
    assert np.allclose(lr_callable(point), expected)

def test_polar_repeat_callable(shape):
    shape = sphere(r=1.0)
    repetitions = 6.0
    pr_shape = shape.polar_repeat(repetitions)
    pr_callable = pr_shape.to_callable()
    angle = 2 * np.pi / repetitions
    point = np.array([[0.3 * np.cos(angle), 0.2, 0.3 * np.sin(angle)]])
    unrotated_point = np.array([[0.3, 0.2, 0]])
    expected = shape.to_callable()(unrotated_point)
    assert np.allclose(pr_callable(point), expected)

def test_elongate_callable(shape):
    h = np.array([0, 0.5, 0])
    e_shape = shape.elongate(h)
    e_callable = e_shape.to_callable()
    point = np.array([[0.1, 0.6, 0.3]])
    p = point
    q = p - np.clip(p, -h, h)
    expected = shape.to_callable()(q)
    assert np.allclose(e_callable(point), expected)