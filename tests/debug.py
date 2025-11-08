import pytest
from sdforge import sphere, Debug
from sdforge.render import NativeRenderer
from unittest.mock import MagicMock

def test_debug_instantiation():
    """Tests the Debug class API."""
    d = Debug('normals')
    assert d.mode == 'normals'

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