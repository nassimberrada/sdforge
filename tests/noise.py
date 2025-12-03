import pytest
from sdforge import sphere, box
from sdforge.api.render import SceneCompiler
from tests.conftest import requires_glsl_validator, HEADLESS_SUPPORTED
import numpy as np

# --- API and Compilation Tests ---

def test_displace_api():
    """Tests the API and GLSL generation for generic displacement."""
    s = sphere(radius=1.0).displace("p.x * 0.1")
    scene_code = SceneCompiler().compile(s)
    assert "opDisplace" in scene_code
    assert "p.x * 0.1" in scene_code

def test_displace_with_mask_glsl():
    """Tests that displacement with mask generates mix logic."""
    mask = box(1.0)
    s = sphere(radius=1.0).displace("0.1", mask=mask, mask_falloff=0.2)
    scene_code = SceneCompiler().compile(s)
    assert "smoothstep" in scene_code
    assert "* (1.0 - smoothstep" in scene_code

def test_displace_by_noise_api():
    """Tests the API and GLSL generation for noise displacement."""
    s = sphere(radius=1.0).displace_by_noise(scale=5.0, strength=0.2)
    scene_code = SceneCompiler().compile(s)
    assert "opDisplace" in scene_code
    assert "snoise" in scene_code
    assert "p * 5.0" in scene_code
    assert "* 0.2" in scene_code

# --- Callable Tests ---

@pytest.mark.skipif(not HEADLESS_SUPPORTED, reason="Requires moderngl.")
def test_displace_by_noise_callable_works(headless_env):
    """
    Ensures that noise displacement works on the GPU backend.
    """
    s = sphere(1.0).displace_by_noise(scale=1.0, strength=0.5)
    try:
        evaluator = s.to_callable()
        points = np.zeros((1, 3))
        res = evaluator(points)
        assert res.shape == (1,)
        assert np.isfinite(res[0])
    except TypeError:
        pytest.fail("DisplaceByNoise.to_callable() raised TypeError, but should work on GPU.")

# --- GLSL Validation Tests ---

NOISE_TEST_CASES = [
    sphere(radius=1.0).displace("sin(p.y * 10.0) * 0.1"),
    sphere(radius=1.0).displace("0.1", mask=box(0.5)),
    sphere(radius=1.0).displace_by_noise(),
    sphere(radius=1.0).displace_by_noise(mask=box(0.5)),
]

@requires_glsl_validator
@pytest.mark.parametrize("sdf_obj", NOISE_TEST_CASES, ids=[repr(o) for o in NOISE_TEST_CASES])
def test_noise_glsl_compiles(validate_glsl, sdf_obj):
    """Tests that the GLSL generated for all noise ops is syntactically valid."""
    scene_code = SceneCompiler().compile(sdf_obj)
    validate_glsl(scene_code)