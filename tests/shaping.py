import pytest
import numpy as np
from sdforge import sphere, box, circle, rectangle, X
from sdforge.render import SceneCompiler
from tests.conftest import requires_glsl_validator

# --- Callable Tests ---

def test_round_callable():
    s_callable = sphere(1.0).round(0.1).to_callable()
    points = np.array([[0.5, 0, 0], [1.1, 0, 0]])
    expected = np.array([-0.6, 0.0])
    assert np.allclose(s_callable(points), expected)

def test_shell_callable():
    s_callable = sphere(1.0).shell(0.1).to_callable()
    points = np.array([[0.5, 0, 0], [1.1, 0, 0]])
    expected = np.array([0.4, 0.0])
    assert np.allclose(s_callable(points), expected)

def test_extrude_callable():
    c_callable = circle(r=1.0).extrude(height=0.5).to_callable()
    points = np.array([[0,0,0], [1,0,0.5], [0,0,1]]) # Inside, on edge, outside
    d = np.linalg.norm(points[:, :2], axis=-1) - 1.0
    w = np.stack([d, np.abs(points[:, 2]) - 0.5], axis=-1)
    expected = np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=-1)
    assert np.allclose(c_callable(points), expected)

def test_revolve_callable():
    prof_callable = rectangle(size=(0.4, 1.0)).translate(X).to_callable()
    rev_callable = rectangle(size=(0.4, 1.0)).translate(X).revolve().to_callable()
    
    points_3d = np.array([[1.2, 0.2, 0], [0.8, -0.4, 0], [1.0, 0.6, 0]])
    points_2d = np.stack([np.linalg.norm(points_3d[:,[0,2]], axis=-1), points_3d[:,1], np.zeros(len(points_3d))], axis=-1)
    expected = prof_callable(points_2d)
    assert np.allclose(rev_callable(points_3d), expected)

# --- Equivalence and Compilation Tests ---

SHAPING_TEST_CASES = [
    box(1.5).round(0.1),
    sphere(1.0).shell(0.05),
    circle(1.0).extrude(0.5),
    rectangle(size=(0.5, 1.0)).translate(X * 1.0).revolve(),
]

@pytest.mark.usefixtures("assert_equivalence")
@pytest.mark.parametrize("sdf_obj", SHAPING_TEST_CASES, ids=[repr(o) for o in SHAPING_TEST_CASES])
def test_shaping_equivalence(assert_equivalence, sdf_obj):
    """Tests numeric equivalence between Python and GLSL for all shaping ops."""
    assert_equivalence(sdf_obj)

@requires_glsl_validator
@pytest.mark.parametrize("sdf_obj", SHAPING_TEST_CASES, ids=[repr(o) for o in SHAPING_TEST_CASES])
def test_shaping_glsl_compiles(validate_glsl, sdf_obj):
    """Tests that the GLSL generated for all shaping ops is syntactically valid."""
    scene_code = SceneCompiler().compile(sdf_obj)
    validate_glsl(scene_code)