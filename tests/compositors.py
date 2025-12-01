import pytest
import numpy as np
from sdforge import sphere, box, SDFNode, Group
from sdforge.api.scene import SceneCompiler
from sdforge.api.core import GLSLContext
from sdforge.api.compositors import Compositor
from sdforge.api.operators import Operator
from examples.compositors import (
    union_example, intersection_example, difference_example,
    smooth_union_example, morphing_example,
)
from tests.conftest import requires_glsl_validator

@pytest.fixture
def shapes():
    return sphere(radius=1.0), box(size=1.5)

def test_union_api_and_callable(shapes):
    s, b = shapes
    u = s.union(b)
    u_op = s | b
    assert isinstance(u, Compositor)
    assert u.op_type == 'union'
    ctx = GLSLContext(compiler=None)
    u.to_glsl(ctx)
    assert "opU(" in "\n".join(ctx.statements)
    ctx_op = GLSLContext(compiler=None)
    u_op.to_glsl(ctx_op)
    assert "\n".join(ctx.statements) == "\n".join(ctx_op.statements)
    u_callable = u.to_callable()
    points = np.array([[0.8, 0, 0], [1.2, 0, 0]])
    expected = np.minimum(s.to_callable()(points), b.to_callable()(points))
    assert np.allclose(u_callable(points), expected)

def test_intersection_api_and_callable(shapes):
    s, b = shapes
    i = s.intersection(b)
    assert isinstance(i, Compositor)
    assert i.op_type == 'intersection'
    ctx = GLSLContext(compiler=None)
    i.to_glsl(ctx)
    assert "opI(" in "\n".join(ctx.statements)
    i_callable = i.to_callable()
    points = np.array([[0.8, 0, 0], [1.2, 0, 0]])
    expected = np.maximum(s.to_callable()(points), b.to_callable()(points))
    assert np.allclose(i_callable(points), expected)

def test_difference_api_and_callable(shapes):
    s, b = shapes
    d = s.difference(b)
    assert isinstance(d, Compositor)
    assert d.op_type == 'difference'
    ctx = GLSLContext(compiler=None)
    d.to_glsl(ctx)
    assert "opS(" in "\n".join(ctx.statements)
    d_callable = d.to_callable()
    points = np.array([[0.8, 0, 0], [1.2, 0, 0]])
    expected = np.maximum(s.to_callable()(points), -b.to_callable()(points))
    assert np.allclose(d_callable(points), expected)

def test_blend_params_api(shapes):
    s, b = shapes
    d1 = s.difference(b, blend=0.1)
    assert d1.blend == 0.1
    assert d1.blend_type == 'smooth'
    d2 = s.difference(b, blend=0.2, blend_type='linear')
    assert d2.blend == 0.2
    assert d2.blend_type == 'linear'

def test_morph_api_and_callable(shapes):
    s, b = shapes
    m = s.morph(b, factor=0.75)
    assert isinstance(m, Compositor)
    assert m.op_type == 'morph'
    ctx = GLSLContext(compiler=None)
    m.to_glsl(ctx)
    assert "opMorph(" in "\n".join(ctx.statements)
    assert "0.75" in "\n".join(ctx.statements)
    m_callable = m.to_callable()
    points = np.array([[0.8, 0, 0], [1.2, 0, 0]])
    dist_a = s.to_callable()(points)
    dist_b = b.to_callable()(points)
    t = 0.75
    expected = (1.0 - t) * dist_a + t * dist_b
    assert np.allclose(m_callable(points), expected)

def test_masked_union_blend(shapes):
    s, b = shapes
    mask = box(0.5)
    u = s.union(b, blend=0.5, mask=mask)
    u_callable = u.to_callable()
    p_out = np.array([[2,2,2]])
    d_out = u_callable(p_out)
    u_sharp = s | b
    d_sharp = u_sharp.to_callable()(p_out)
    assert np.allclose(d_out, d_sharp, atol=1e-4)

def test_group_acts_as_union(shapes):
    s, b = shapes
    g = Group(s, b)
    assert isinstance(g, Compositor)
    u = Compositor([s, b], op_type='union')
    points = np.random.rand(100, 3)
    assert np.allclose(g.to_callable()(points), u.to_callable()(points))

def test_empty_group():
    g = Group()
    points = np.array([[0., 0., 0.], [10., 20., 30.]])
    distances = g.to_callable()(points)
    assert np.all(distances > 1e8)

def test_transform_propagation(shapes):
    s, b = shapes
    g = Group(s, b)
    offset = (1, 2, 3)
    translated_group = g.translate(offset)
    assert isinstance(translated_group, Operator)
    assert translated_group.func_name == 'opTranslate'
    assert translated_group.child == g

OPERATION_TEST_CASES = [
    sphere(1.0) | box(1.5), sphere(1.0) & box(1.5), sphere(1.0) - box(1.5),
    sphere(1.0).union(box(1.5), blend=0.2),
    sphere(1.0).intersection(box(1.5), blend=0.2, blend_type='linear'),
    sphere(1.0).morph(box(1.5), factor=0.5),
    sphere(1.0).union(box(1.5), blend=0.5, mask=box(0.5)),
]

@pytest.mark.usefixtures("assert_equivalence")
@pytest.mark.parametrize("sdf_obj", OPERATION_TEST_CASES, ids=[repr(o) for o in OPERATION_TEST_CASES])
def test_operation_equivalence(assert_equivalence, sdf_obj):
    assert_equivalence(sdf_obj)

@requires_glsl_validator
@pytest.mark.parametrize("sdf_obj", OPERATION_TEST_CASES, ids=[repr(o) for o in OPERATION_TEST_CASES])
def test_operation_glsl_compiles(validate_glsl, sdf_obj):
    scene_code = SceneCompiler().compile(sdf_obj)
    validate_glsl(scene_code)

EXAMPLE_TEST_CASES = [
    (union_example, Compositor), (intersection_example, Compositor),
    (difference_example, Compositor), (smooth_union_example, Compositor),
    (morphing_example, Compositor),
]

@pytest.mark.parametrize("example_func, expected_class", EXAMPLE_TEST_CASES, ids=[f[0].__name__ for f in EXAMPLE_TEST_CASES])
def test_operation_example_runs(example_func, expected_class):
    scene = example_func()
    assert isinstance(scene, SDFNode)
    assert isinstance(scene, expected_class)