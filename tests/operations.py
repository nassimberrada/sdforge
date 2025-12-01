import pytest
import numpy as np
from sdforge import sphere, box, SDFNode
from sdforge.api.render import SceneCompiler
from sdforge.api.core import GLSLContext
from tests.conftest import requires_glsl_validator
from examples.operations import (
    union_example,
    intersection_example,
    difference_example,
    smooth_union_example,
    morphing_example,
)
from sdforge.api.operations import Union, Intersection, Difference, Morph

@pytest.fixture
def shapes():
    """Provides two basic shapes for operation tests."""
    return sphere(radius=1.0), box(size=1.5)

# --- API and Callable Tests ---

def test_union_api_and_callable(shapes):
    s, b = shapes
    u = s.union(b)
    u_op = s | b

    # Check GLSL generation
    ctx = GLSLContext(compiler=None)
    u.to_glsl(ctx)
    generated_code = "\n".join(ctx.statements)
    assert "opU(" in generated_code

    ctx_op = GLSLContext(compiler=None)
    u_op.to_glsl(ctx_op)
    generated_code_op = "\n".join(ctx_op.statements)
    assert generated_code == generated_code_op

    # Check Python callable
    u_callable = u.to_callable()
    points = np.array([[0.8, 0, 0], [1.2, 0, 0]])
    expected = np.minimum(s.to_callable()(points), b.to_callable()(points))
    assert np.allclose(u_callable(points), expected)

def test_intersection_api_and_callable(shapes):
    s, b = shapes
    i = s.intersection(b)
    i_op = s & b

    # Check GLSL generation
    ctx = GLSLContext(compiler=None)
    i.to_glsl(ctx)
    generated_code = "\n".join(ctx.statements)
    assert "opI(" in generated_code

    ctx_op = GLSLContext(compiler=None)
    i_op.to_glsl(ctx_op)
    generated_code_op = "\n".join(ctx_op.statements)
    assert generated_code == generated_code_op

    # Check Python callable
    i_callable = i.to_callable()
    points = np.array([[0.8, 0, 0], [1.2, 0, 0]])
    expected = np.maximum(s.to_callable()(points), b.to_callable()(points))
    assert np.allclose(i_callable(points), expected)

def test_difference_api_and_callable(shapes):
    s, b = shapes
    d = s.difference(b)
    d_op = s - b

    # Check GLSL generation
    ctx = GLSLContext(compiler=None)
    d.to_glsl(ctx)
    generated_code = "\n".join(ctx.statements)
    assert "opS(" in generated_code

    ctx_op = GLSLContext(compiler=None)
    d_op.to_glsl(ctx_op)
    generated_code_op = "\n".join(ctx_op.statements)
    assert generated_code == generated_code_op

    # Check Python callable
    d_callable = d.to_callable()
    points = np.array([[0.8, 0, 0], [1.2, 0, 0]])
    expected = np.maximum(s.to_callable()(points), -b.to_callable()(points))
    assert np.allclose(d_callable(points), expected)

def test_blend_params_api(shapes):
    """Tests that using blend and blend_type sets properties correctly."""
    s, b = shapes
    # Test smooth (default)
    d1 = s.difference(b, blend=0.1)
    assert isinstance(d1, Difference)
    assert d1.blend == 0.1
    assert d1.blend_type == 'smooth'

    # Test linear
    d2 = s.difference(b, blend=0.2, blend_type='linear')
    assert isinstance(d2, Difference)
    assert d2.blend == 0.2
    assert d2.blend_type == 'linear'

def test_morph_api_and_callable(shapes):
    s, b = shapes
    m = s.morph(b, factor=0.75)

    # Check GLSL generation
    ctx = GLSLContext(compiler=None)
    m.to_glsl(ctx)
    generated_code = "\n".join(ctx.statements)
    assert "opMorph(" in generated_code
    # Check that factor is formatted correctly
    assert "0.75" in generated_code

    # Check Python callable
    m_callable = m.to_callable()
    points = np.array([[0.8, 0, 0], [1.2, 0, 0]])

    dist_a = s.to_callable()(points)
    dist_b = b.to_callable()(points)
    t = 0.75
    expected = (1.0 - t) * dist_a + t * dist_b

    assert np.allclose(m_callable(points), expected)

def test_masked_union_blend(shapes):
    """Tests that blending can be localized with a mask."""
    s, b = shapes
    # Union with blend=0.5, but masked to only apply inside a small box.
    mask = box(0.5) # Center 0.5x0.5x0.5
    u = s.union(b, blend=0.5, mask=mask)
    u_callable = u.to_callable()
    
    # Point near intersection at origin (inside mask).
    # Should reflect blended distance.
    # At origin, s=-1, b=-0.75. 
    # Smooth union of -1, -0.75 with k=0.5:
    # h = clamp(0.5 + 0.5*(-0.75 - -1)/0.5, 0, 1) = clamp(0.5 + 0.25/0.5) = clamp(1.0) = 1.0?
    # Wait, simple min is -1. sMin pulls it further negative.
    
    # Point outside mask (2,2,2). Blend should be effectively 0.
    # Distances are positive here.
    p_out = np.array([[2,2,2]])
    d_out = u_callable(p_out)
    
    # Reference sharp union
    u_sharp = s | b
    d_sharp = u_sharp.to_callable()(p_out)
    
    assert np.allclose(d_out, d_sharp, atol=1e-4)

# --- Equivalence and Compilation Tests ---

OPERATION_TEST_CASES = [
    sphere(radius=1.0) | box(size=1.5),
    sphere(radius=1.0) & box(size=1.5),
    sphere(radius=1.0) - box(size=1.5),
    # Smooth blend
    sphere(radius=1.0).union(box(size=1.5), blend=0.2),
    sphere(radius=1.0).intersection(box(size=1.5), blend=0.2),
    sphere(radius=1.0).difference(box(size=1.5), blend=0.2),
    # linear blend
    sphere(radius=1.0).union(box(size=1.5), blend=0.2, blend_type='linear'),
    sphere(radius=1.0).intersection(box(size=1.5), blend=0.2, blend_type='linear'),
    sphere(radius=1.0).difference(box(size=1.5), blend=0.2, blend_type='linear'),
    # Morphing
    sphere(radius=1.0).morph(box(size=1.5), factor=0.5),
    sphere(radius=1.0).morph(box(size=1.5), factor=0.0),
    sphere(radius=1.0).morph(box(size=1.5), factor=1.0),
    # Masked Operations
    sphere(1.0).union(box(1.5), blend=0.5, mask=box(0.5)),
    sphere(1.0).difference(box(1.5), blend=0.5, mask=box(0.5)),
    sphere(1.0).morph(box(1.5), factor=0.5, mask=box(0.5)),
    # Test chaining
    (sphere(radius=1.0) | box(size=0.8)) - sphere(radius=0.5)
]

@pytest.mark.usefixtures("assert_equivalence")
@pytest.mark.parametrize("sdf_obj", OPERATION_TEST_CASES, ids=[repr(o) for o in OPERATION_TEST_CASES])
def test_operation_equivalence(assert_equivalence, sdf_obj):
    """Tests numeric equivalence between Python and GLSL for all operations."""
    assert_equivalence(sdf_obj)

@requires_glsl_validator
@pytest.mark.parametrize("sdf_obj", OPERATION_TEST_CASES, ids=[repr(o) for o in OPERATION_TEST_CASES])
def test_operation_glsl_compiles(validate_glsl, sdf_obj):
    """Tests that the GLSL generated for all operations is syntactically valid."""
    scene_code = SceneCompiler().compile(sdf_obj)
    validate_glsl(scene_code)

# --- Example File Tests ---

EXAMPLE_TEST_CASES = [
    (union_example, Union),
    (intersection_example, Intersection),
    (difference_example, Difference),
    (smooth_union_example, Union),
    (morphing_example, Morph),
]

@pytest.mark.parametrize("example_func, expected_class", EXAMPLE_TEST_CASES, ids=[f[0].__name__ for f in EXAMPLE_TEST_CASES])
def test_operation_example_runs(example_func, expected_class):
    """
    Tests that the operation example functions from the examples file
    run without errors and return a valid SDFNode of the correct type.
    """
    scene = example_func()
    assert isinstance(scene, SDFNode)
    assert isinstance(scene, expected_class)