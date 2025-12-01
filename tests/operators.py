import pytest
import numpy as np
from sdforge import sphere, box, circle, rectangle, X, Y, Z
from sdforge.api.operators import Operator
from sdforge.api.scene import SceneCompiler
from sdforge.api.core import GLSLContext
from tests.conftest import requires_glsl_validator

@pytest.fixture
def shape():
    return box(size=(1, 2, 3))

def test_translate_api_and_callable(shape):
    offset = np.array([1, 2, 3])
    t_shape = shape.translate(offset)
    t_op = shape + offset
    assert isinstance(t_shape, Operator)
    assert t_shape.op_type == 'transform'
    assert t_shape.func_name == 'opTranslate'
    t_callable = t_shape.to_callable()
    point = np.array([[1.1, 2.2, 3.3]])
    expected = shape.to_callable()(point - offset)
    assert np.allclose(t_callable(point), expected)
    assert np.allclose(t_op.to_callable()(point), expected)

def test_masked_translate_callable(shape):
    offset = np.array([10, 0, 0]); mask = sphere(1.0) 
    t_callable = shape.translate(offset, mask=mask, mask_falloff=0.0).to_callable()
    p_in = np.array([[0,0,0]])
    d_in = t_callable(p_in)
    assert d_in > 5.0 
    p_out = np.array([[5,0,0]])
    d_out = t_callable(p_out)
    assert np.isclose(d_out, 4.5, atol=0.1)

def test_scale_api_and_callable(shape):
    factor = 2.0
    s_shape = shape.scale(factor)
    s_op1, s_op2 = shape * factor, factor * shape
    assert isinstance(s_shape, Operator)
    assert s_shape.func_name == 'opScale'
    assert s_shape.dist_correction == 2.0
    s_callable = s_shape.to_callable()
    point = np.array([[0.6, 1.2, 1.8]])
    expected = shape.to_callable()(point / factor) * factor
    assert np.allclose(s_callable(point), expected)
    assert np.allclose(s_op1.to_callable()(point), expected)
    assert np.allclose(s_op2.to_callable()(point), expected)

def test_rotate_api_and_callable(shape):
    angle = np.pi / 2
    r_shape = shape.rotate(Z, angle)
    assert isinstance(r_shape, Operator)
    assert r_shape.func_name == 'opRotateZ'
    r_callable = r_shape.to_callable()
    point = np.array([[2.5, 0.6, 0]])
    c, s = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    expected = shape.to_callable()(point @ rot_matrix.T)
    assert np.allclose(r_callable(point), expected)

def test_orient_callable(shape):
    o_callable = shape.orient(X).to_callable()
    point = np.array([[3.1, 2.1, 1.1]])
    expected = shape.to_callable()(point[:, [2, 1, 0]])
    assert np.allclose(o_callable(point), expected)

def test_twist_callable(shape):
    k = 5.0; t_callable = shape.twist(strength=k).to_callable()
    point = np.array([[0.1, 0.2, 0.3]]); p = point
    c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
    x_new, z_new = p[:,0]*c - p[:,2]*s, p[:,0]*s + p[:,2]*c
    q = np.stack([x_new, p[:,1], z_new], axis=-1)
    expected = shape.to_callable()(q)
    assert np.allclose(t_callable(point), expected)

def test_masked_twist_callable(shape):
    k = 5.0; mask = box(1.0).translate((0, 10, 0))
    t_callable = shape.twist(strength=k, mask=mask).to_callable()
    assert np.isclose(t_callable(np.array([[0.5, 0, 0]])), 0.0, atol=1e-4)
    assert abs(t_callable(np.array([[0.5, 10, 0]]))) > 0.01

def test_bend_callable(shape):
    k = 0.5; b_callable = shape.bend(Y, curvature=k).to_callable()
    point = np.array([[0.1, 0.2, 0.3]]); p = point
    c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
    x_new, z_new = c * p[:,0] - s * p[:,2], s * p[:,0] + c * p[:,2]
    q = np.stack([x_new, p[:,1], z_new], axis=-1)
    expected = shape.to_callable()(q)
    assert np.allclose(b_callable(point), expected)

def test_mirror_callable(shape):
    m_callable = shape.mirror(X | Z).to_callable()
    point = np.array([[-0.1, 0.2, -0.3]])
    expected = shape.to_callable()(np.abs(point))
    assert np.allclose(m_callable(point), expected)

def test_warp_api_structure():
    s = sphere(1.0).warp(frequency=2.0, strength=0.5)
    assert isinstance(s, Operator)
    assert s.func_name == 'opWarp'
    assert len(s.params) == 2

def test_warp_glsl_generation():
    s = sphere(1.0).warp(frequency=3.0, strength=1.0)
    scene_code = SceneCompiler().compile(s)
    assert "opWarp" in scene_code
    assert "snoiseVec3" in scene_code 

def test_warp_callable_raises_error():
    s = sphere(1.0).warp(frequency=2.0, strength=0.5)
    with pytest.raises(TypeError, match="GPU-only"):
        s.to_callable()

def test_round_callable():
    s_callable = sphere(radius=1.0).round(0.1).to_callable()
    points = np.array([[0.5, 0, 0], [1.1, 0, 0]])
    expected = np.array([-0.6, 0.0])
    assert np.allclose(s_callable(points), expected)

def test_masked_round_callable():
    s = sphere(1.0); mask = box(2.0).translate((2.0, 0, 0)) 
    s_callable = s.round(0.1, mask=mask, mask_falloff=0.0).to_callable()
    assert np.isclose(s_callable(np.array([[-1.0, 0, 0]])), 0.0, atol=1e-4)
    mask2 = box(1.0)
    s_callable2 = s.round(0.1, mask=mask2, mask_falloff=0.0).to_callable()
    assert np.isclose(s_callable2(np.array([[0,0,0]])), -1.1, atol=1e-4)

def test_shell_callable():
    s_callable = sphere(radius=1.0).shell(0.1).to_callable()
    points = np.array([[0.5, 0, 0], [1.1, 0, 0]])
    expected = np.array([0.4, 0.0])
    assert np.allclose(s_callable(points), expected)

def test_extrude_callable():
    c_callable = circle(radius=1.0).extrude(height=0.5).to_callable()
    points = np.array([[0,0,0], [1,0,0.5], [0,0,1]]) 
    d = np.linalg.norm(points[:, :2], axis=-1) - 1.0
    w = np.stack([d, np.abs(points[:, 2]) - 0.5], axis=-1)
    expected = np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=-1)
    assert np.allclose(c_callable(points), expected)

def test_revolve_callable():
    rev_callable = rectangle(size=(0.4, 1.0)).translate(X).revolve().to_callable()
    prof_callable = rectangle(size=(0.4, 1.0)).translate(X).to_profile_callable()
    points_3d = np.array([[1.2, 0.2, 0], [0.8, -0.4, 0], [1.0, 0.6, 0]])
    points_2d = np.stack([np.linalg.norm(points_3d[:,[0,2]], axis=-1), points_3d[:,1], np.zeros(len(points_3d))], axis=-1)
    expected = prof_callable(points_2d)
    assert np.allclose(rev_callable(points_3d), expected)

def test_displace_api():
    s = sphere(radius=1.0).displace("p.x * 0.1")
    scene_code = SceneCompiler().compile(s)
    assert "opDisplace" in scene_code
    assert "p.x * 0.1" in scene_code

def test_displace_by_noise_api():
    s = sphere(radius=1.0).displace_by_noise(scale=5.0, strength=0.2)
    scene_code = SceneCompiler().compile(s)
    assert "opDisplace" in scene_code
    assert "snoise" in scene_code

def test_displace_fails_callable():
    s = sphere(radius=1.0).displace("p.x * 0.1")
    with pytest.raises(TypeError, match="GPU-only"): s.to_callable()

def test_material_api():
    s = sphere(radius=1.0)
    colored_s = s.color(1.0, 0.5, 0.2)
    assert isinstance(colored_s, Operator)
    assert colored_s.op_type == 'material'
    assert colored_s.child == s
    assert colored_s.params[0] == (1.0, 0.5, 0.2)

def test_collect_materials():
    s1 = sphere(1.0).color(1, 0, 0)
    s2 = sphere(0.5).color(0, 1, 0)
    b1 = box(1.0).color(1, 0, 0) 
    scene = (s1 | s2) - b1
    materials = []
    scene._collect_materials(materials)
    assert len(materials) == 2
    assert s1.material_id == 0
    assert s2.material_id == 1
    assert b1.material_id == 0

def test_material_glsl_generation():
    s = sphere(1.0)
    mat = s.color(1, 0, 0)
    mat.material_id = 5 
    ctx = GLSLContext(compiler=None)
    result_var = mat.to_glsl(ctx)
    assert f"vec4 {result_var} = vec4(var_0.x, 5.0, var_0.zw);" in "\n".join(ctx.statements)

OPERATOR_TEST_CASES = [
    sphere(0.8).translate((0.5, -0.2, 0.1)),
    box(1.0).scale(2.0),
    sphere(0.8).rotate(X, np.pi / 4),
    sphere(0.8).twist(strength=5.0),
    box(1.0).bend(Y, curvature=0.5),
    sphere(0.5).repeat((2, 2, 0)),
    box(0.4).mirror(X | Y),
    box(1.0).round(0.1),
    sphere(1.0).shell(0.05),
    circle(1.0).extrude(0.5),
    rectangle((0.5, 1.0)).translate(X).revolve(),
    sphere(1.0).displace("0.1", mask=box(0.5)),
    sphere(1.0).displace_by_noise(),
]

@pytest.mark.usefixtures("assert_equivalence")
@pytest.mark.parametrize("sdf_obj", OPERATOR_TEST_CASES, ids=[repr(o) for o in OPERATOR_TEST_CASES])
def test_operator_equivalence(assert_equivalence, sdf_obj):
    if isinstance(sdf_obj, Operator):
        if "Displace" in sdf_obj.func_name or "Warp" in sdf_obj.func_name: return
    assert_equivalence(sdf_obj)

@requires_glsl_validator
@pytest.mark.parametrize("sdf_obj", OPERATOR_TEST_CASES, ids=[repr(o) for o in OPERATOR_TEST_CASES])
def test_operator_glsl_compiles(validate_glsl, sdf_obj):
    scene_code = SceneCompiler().compile(sdf_obj)
    validate_glsl(scene_code, sdf_obj)