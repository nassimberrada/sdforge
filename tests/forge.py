import pytest
import numpy as np
from sdforge import Forge, sphere
from sdforge.api.render import SceneCompiler
from tests.conftest import requires_glsl_validator, HEADLESS_SUPPORTED

# --- API and Compilation Tests ---

def test_forge_api():
    """Tests the basic API and GLSL generation for Forge."""
    f = Forge("length(p) - 1.0")
    scene_code = SceneCompiler().compile(f)
    assert f.unique_id in scene_code
    assert "length(p) - 1.0" in scene_code
    assert f"vec4 var_0 = vec4({f.unique_id}(p), -1.0, 0.0, 0.0);" in scene_code

def test_forge_with_uniforms_api():
    """Tests GLSL generation for Forge with uniforms."""
    uniforms = {'u_radius': 1.5, 'u_offset': 0.1}
    f = Forge("length(p) - u_radius + u_offset", uniforms=uniforms)
    scene_code = SceneCompiler().compile(f)
    
    assert f"in float u_radius, in float u_offset" in scene_code
    assert f"vec4 var_0 = vec4({f.unique_id}(p, u_radius, u_offset), -1.0, 0.0, 0.0);" in scene_code

# --- Callable Tests ---

@pytest.mark.skipif(not HEADLESS_SUPPORTED, reason="Requires moderngl.")
def test_forge_callable_equivalence(headless_env):
    """
    Compares the Forge GPU callable to a standard Primitive GPU callable.
    """
    radius = 1.2
    forge_sphere = Forge(f"length(p) - {radius}")
    native_sphere = sphere(radius)
    
    forge_callable = forge_sphere.to_callable()
    native_callable = native_sphere.to_callable()
    
    points = (np.random.rand(1024, 3) * 4 - 2).astype('f4')
    
    forge_distances = forge_callable(points)
    native_distances = native_callable(points)
    
    assert np.allclose(forge_distances, native_distances, atol=1e-5)

@pytest.mark.skipif(not HEADLESS_SUPPORTED, reason="Requires moderngl.")
def test_forge_callable_with_uniforms(headless_env):
    """Forge with uniforms should now WORK on the GPU."""
    f = Forge("length(p) - u_radius", uniforms={'u_radius': 1.0})
    try:
        evaluator = f.to_callable()
    except Exception as e:
        pytest.fail(f"Forge with uniforms failed to compile/run: {e}")
    
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype='f4')
    dists = evaluator(points)
    # Expected: dist(0) = 0 - 1.0 = -1.0. dist(2) = 2.0 - 1.0 = 1.0.
    assert np.allclose(dists, [-1.0, 1.0], atol=1e-5)

# --- GLSL Validation Tests ---

forge_with_deps = Forge("sdBox(p, vec3(0.5))", uniforms={})
forge_with_deps.glsl_dependencies.add("primitives")

FORGE_TEST_CASES = [
    Forge("length(p) - 1.0"),
    forge_with_deps,
    Forge("length(p) - u_radius", uniforms={'u_radius': 1.2}),
]

@requires_glsl_validator
@pytest.mark.parametrize("sdf_obj", FORGE_TEST_CASES, ids=[repr(o) for o in FORGE_TEST_CASES])
def test_forge_glsl_compiles(validate_glsl, sdf_obj):
    """Tests that the GLSL generated for Forge objects is syntactically valid."""
    scene_code = SceneCompiler().compile(sdf_obj)
    validate_glsl(scene_code, sdf_obj)