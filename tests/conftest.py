import pytest
import numpy as np
from sdforge.api.render import SceneCompiler
import os
import shutil
import subprocess
import tempfile

# Dependency checks
try:
    import moderngl
    import glfw
    HEADLESS_SUPPORTED = True
except ImportError:
    HEADLESS_SUPPORTED = False

GLSL_VALIDATOR = shutil.which("glslangValidator")
SKIP_GLSL = os.environ.get("SKIP_GLSL", "") == "1"

requires_glsl_validator = pytest.mark.skipif(
    not GLSL_VALIDATOR or SKIP_GLSL,
    reason="Requires glslangValidator."
)

@pytest.fixture(scope="session")
def headless_env():
    """Ensures a headless context is available for GPU evaluator tests."""
    if not HEADLESS_SUPPORTED:
        pytest.skip("moderngl/glfw not installed.")
    
    # We rely on api.core.HeadlessContext to handle setup/teardown.
    # Just asserting it initializes correctly here.
    try:
        from sdforge.api.core import HeadlessContext
        ctx = HeadlessContext.get()
        yield ctx
    except Exception as e:
        pytest.skip(f"Failed to init headless context: {e}")

@pytest.fixture
def assert_equivalence(headless_env):
    """
    Deprecated: Previously checked NumPy vs GLSL.
    Now, since everything is GLSL, this simply runs the GPU evaluator
    to ensure the GLSL compiles and returns valid numbers (no NaNs).
    """
    def _asserter(sdf_obj):
        points = (np.random.rand(100, 3) * 4 - 2).astype('f4')
        try:
            dist = sdf_obj.to_callable()(points)
            assert not np.any(np.isnan(dist)), "Evaluator returned NaNs"
            assert dist.shape == (100,)
        except Exception as e:
            pytest.fail(f"GPU evaluation failed: {e}")

    return _asserter

@pytest.fixture(scope="session")
def validate_glsl():
    def _validator(scene_code: str, sdf_obj=None):
        uniforms = {}
        if sdf_obj: sdf_obj._collect_uniforms(uniforms)
        uniform_declarations = "\n".join([f"uniform float {name};" for name in uniforms.keys()])

        shader = f"""
        #version 430 core
        out vec4 f_color;
        {uniform_declarations}
        {scene_code}
        void main() {{
            vec3 p = vec3(0.0);
            f_color = Scene(p);
        }}
        """
        with tempfile.NamedTemporaryFile(suffix=".frag", mode="w", delete=True) as f:
            f.write(shader)
            f.flush()
            result = subprocess.run([GLSL_VALIDATOR, "-S", "frag", f.name], capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"GLSL Validation Failed:\n{result.stderr}\nSOURCE:\n{scene_code}")
    return _validator