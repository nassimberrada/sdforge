import pytest
from sdforge import sphere, Debug
from sdforge.render import NativeRenderer
from unittest.mock import MagicMock

def test_debug_instantiation():
    """Tests the Debug class API."""
    d = Debug('normals')
    assert d.mode == 'normals'

    d_slice = Debug('slice', plane='XZ', slice_height=1.5, view_scale=10.0)
    assert d_slice.mode == 'slice'
    assert d_slice.plane == 'xz' # Lowercase conversion
    assert d_slice.slice_height == 1.5
    assert d_slice.view_scale == 10.0

def test_render_with_debug_normals():
    """Tests that the 'normals' debug mode generates the correct GLSL."""
    s = sphere()
    renderer = NativeRenderer(s, debug=Debug('normals'))

    # Mock the context since we are not creating a real GL window
    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx

    renderer._compile_shader()

    # Get the fragment shader source from the call to the mock program
    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')

    assert "color = debugNormals(normal);" in fragment_shader
    assert "ambientOcclusion(p, normal" not in fragment_shader

def test_render_with_debug_steps():
    """Tests that the 'steps' debug mode generates the correct GLSL."""
    s = sphere()
    renderer = NativeRenderer(s, debug=Debug('steps'))

    # Mock the context
    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx

    renderer._compile_shader()

    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')

    assert "color = debugSteps(hit.z, 100.0);" in fragment_shader

def test_render_with_slice_debug():
    """Tests that the 'slice' debug mode generates specialized GLSL."""
    s = sphere()
    # XY plane at Z=0.5, scale 4.0
    debug = Debug('slice', plane='xy', slice_height=0.5, view_scale=4.0)
    renderer = NativeRenderer(s, debug=debug)

    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx
    renderer._compile_shader()

    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')

    # Check that raymarching is NOT used
    assert "raymarch(ro, rd)" not in fragment_shader
    assert "cameraStatic" not in fragment_shader

    # Check that slice mapping logic is present
    # st * (scale/2) = st * 2.0
    assert "vec3 p = vec3(st * 2.0, 0.5)" in fragment_shader

    # Check that debugDistanceField is called
    assert "debugDistanceField(d)" in fragment_shader

def test_render_with_slice_debug_xz_plane():
    """Tests correct coordinate mapping for XZ slice."""
    s = sphere()
    debug = Debug('slice', plane='xz', slice_height=1.0, view_scale=2.0)
    renderer = NativeRenderer(s, debug=debug)

    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx
    renderer._compile_shader()

    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')

    # Y should be fixed at height=1.0. X and Z map from st.
    assert "vec3 p = vec3(st.x * 1.0, 1.0, st.y * 1.0)" in fragment_shader

def test_render_with_invalid_debug_mode(capsys):
    """Tests that an invalid debug mode falls back to standard lighting."""
    s = sphere()
    renderer = NativeRenderer(s, debug=Debug('invalid_mode'))

    # Mock the context
    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx

    renderer._compile_shader()

    captured = capsys.readouterr()
    assert "WARNING: Unknown debug mode 'invalid_mode'" in captured.out

    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')

    # Should fall back to standard lighting, which includes a call to ambientOcclusion
    assert "ambientOcclusion(p, normal" in fragment_shader
    assert "debugNormals(normal)" not in fragment_shader