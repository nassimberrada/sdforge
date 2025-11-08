import pytest
import numpy as np
from sdforge.render import SceneCompiler
import os
import shutil
import subprocess
import tempfile
import re

# --- Dependency Check for GLSL Validator ---
GLSL_VALIDATOR = shutil.which("glslangValidator")
SKIP_GLSL = os.environ.get("SKIP_GLSL", "") == "1"

# Mark for skipping tests if the validator is not available
requires_glsl_validator = pytest.mark.skipif(
    not GLSL_VALIDATOR or SKIP_GLSL,
    reason="Requires glslangValidator on the PATH. Set SKIP_GLSL=1 to disable."
)

# --- Dependency Check for Headless Rendering ---
try:
    import moderngl
    import glfw
    HEADLESS_SUPPORTED = True
except ImportError:
    HEADLESS_SUPPORTED = False

# --- Shared Test Data ---
np.random.seed(42)
TEST_POINTS = (np.random.rand(4096, 3) * 4 - 2).astype('f4')

@pytest.fixture(scope="session")
def mgl_context():
    """A session-scoped fixture that provides a headless ModernGL context."""
    if not HEADLESS_SUPPORTED:
        pytest.skip("This test requires moderngl and glfw for headless GLSL evaluation.")
    
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    
    glfw.window_hint(glfw.VISIBLE, False)
    win = glfw.create_window(1, 1, "pytest-headless", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context(require=430)
    
    yield ctx
    glfw.terminate()

@pytest.fixture
def assert_equivalence(mgl_context):
    """
    Provides a helper to assert that an SDF's NumPy and GLSL versions are equivalent.
    """
    def _get_glsl_distances(sdf_obj, points):
        scene_code = SceneCompiler().compile(sdf_obj)
        
        compute_shader_src = f"""
        #version 430
        layout(local_size_x=256) in;
        layout(std430, binding=0) buffer points {{ vec3 p_in[]; }};
        layout(std430, binding=1) buffer distances {{ float d_out[]; }};

        {scene_code}

        void main() {{
            uint gid = gl_GlobalInvocationID.x;
            if (gid >= {len(points)}) return;
            d_out[gid] = Scene(p_in[gid]).x;
        }}
        """
        try:
            compute_shader = mgl_context.compute_shader(compute_shader_src)
        except Exception as e:
            pytest.fail(f"GLSL compilation failed for equivalence test:\n{e}")

        padded_points = np.zeros((points.shape[0], 4), dtype='f4')
        padded_points[:, :3] = points
        point_buffer = mgl_context.buffer(padded_points.tobytes())
        
        dist_buffer = mgl_context.buffer(reserve=points.shape[0] * 4)
        
        point_buffer.bind_to_storage_buffer(0)
        dist_buffer.bind_to_storage_buffer(1)
        
        group_size = (points.shape[0] + 255) // 256
        compute_shader.run(group_x=group_size)
        
        return np.frombuffer(dist_buffer.read(), dtype='f4')

    def _asserter(sdf_obj):
        numpy_callable = sdf_obj.to_callable()
        numpy_distances = numpy_callable(TEST_POINTS)
        glsl_distances = _get_glsl_distances(sdf_obj, TEST_POINTS)
        
        assert np.allclose(numpy_distances, glsl_distances, atol=1e-4), \
            f"Mismatch between NumPy and GLSL for {type(sdf_obj).__name__}"

    return _asserter


@pytest.fixture(scope="session")
def validate_glsl():
    """
    Provides a function to validate a GLSL Scene function using glslangValidator.
    """
    def _validator(scene_code: str, sdf_obj=None):
        """
        Wraps scene_code in a minimal fragment shader and runs the validator.
        Raises an AssertionError on failure.
        """
        # Collect uniforms from the sdf_obj to declare them in the test shader
        uniforms = {}
        if sdf_obj:
            sdf_obj._collect_uniforms(uniforms)
        
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
            
            result = subprocess.run(
                [GLSL_VALIDATOR, "-S", "frag", f.name],
                capture_output=True, text=True
            )

        if result.returncode != 0:
            raise AssertionError(
                f"glslangValidator failed (exit {result.returncode}).\n"
                f"STDERR:\n{result.stderr}\n\n"
                f"------ GLSL Scene Code ------\n{scene_code}"
            )
            
    return _validator