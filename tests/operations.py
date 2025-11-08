import pytest
import numpy as np
from sdforge import sphere, box, SDFNode
from sdforge.render import SceneCompiler
from sdforge.core import GLSLContext
from tests.conftest import requires_glsl_validator
from examples.operations import (
    union_example,
    intersection_example,
    difference_example,
    smooth_union_example,
)
from sdforge.api.operations import Union, Intersection, Difference

@pytest.fixture
def shapes():
    """Provides two basic shapes for operation tests."""
    return sphere(r=1.0), box(size=1.5)

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

# --- Equivalence and Compilation Tests ---

OPERATION_TEST_CASES = [
    sphere(r=1.0) | box(size=1.5),
    sphere(r=1.0) & box(size=1.5),
    sphere(r=1.0) - box(size=1.5),
    sphere(r=1.0).union(box(size=1.5), k=0.2),
    sphere(r=1.0).intersection(box(size=1.5), k=0.2),
    sphere(r=1.0).difference(box(size=1.5), k=0.2),
    # Test chaining
    (sphere(r=1.0) | box(size=0.8)) - sphere(0.5)
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
    (smooth_union_example, Union), # smooth_union is still a Union node
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