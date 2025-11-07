import pytest
from sdforge import *
import sys
import os
from unittest.mock import MagicMock, patch
from sdforge.render import assemble_shader_code, NativeRenderer, FileSystemEventHandler

def test_camera_instantiation():
    cam_static = Camera(position=(1, 2, 3), target=(0, 0, 0), zoom=1.5)
    assert cam_static.position == (1, 2, 3)
    assert cam_static.zoom == 1.5

    cam_animated = Camera(
        position=("sin(u_time)", "2.0", "cos(u_time)"),
        target=("0.1", "0.2", "0.3")
    )
    assert cam_animated.position[0] == "sin(u_time)"

def test_light_instantiation():
    light_static = Light(position=(5, 5, 5), ambient_strength=0.2, shadow_softness=16.0)
    assert light_static.ambient_strength == 0.2

    light_animated = Light(
        position=("8.0 * sin(u_time)", "5.0", "8.0 * cos(u_time)"),
        shadow_softness="12.0 + 10.0 * sin(u_time)"
    )
    assert light_animated.shadow_softness == "12.0 + 10.0 * sin(u_time)"

def test_shader_assembly():
    s = sphere(1.2).color(1, 0, 0)
    b = box(1.5).color(0, 1, 0).translate(X * 0.8)
    c = cylinder(0.5)
    
    scene_obj = (s | b) - c.twist(4.0)
    
    custom_shape = Forge("return length(p.xy) - 0.2;")
    final_scene = scene_obj & custom_shape

    try:
        shader_code = assemble_shader_code(final_scene)
        
        assert "vec4 Scene(in vec3 p)" in shader_code
        assert "sdSphere" in shader_code
        assert "sdBox" in shader_code
        assert "opU" in shader_code
        assert "opTwist" in shader_code
        assert "forge_func_" in shader_code
        
    except Exception as e:
        pytest.fail(f"Shader assembly failed with an exception: {e}")

def test_shader_assembly_with_invalid_glsl():
    # This tests that the Python-side assembly does not crash on bad GLSL.
    # The GLSL compiler itself will catch the syntax error at render time.
    invalid_shape = Forge("return length(p) - 1.0 // missing semicolon")
    try:
        shader_code = assemble_shader_code(invalid_shape)
        assert "missing semicolon" in shader_code
    except Exception as e:
        pytest.fail(f"Shader assembly with invalid GLSL failed: {e}")

def test_hot_reloading_triggers_reload():
    mock_renderer = MagicMock(spec=NativeRenderer)
    mock_renderer.script_path = os.path.abspath("my_script.py")
    mock_renderer.reload_pending = False

    class ChangeHandler(FileSystemEventHandler):
        def __init__(self, renderer_instance): self.renderer = renderer_instance
        def on_modified(self, event):
            if event.src_path == self.renderer.script_path:
                self.renderer.reload_pending = True
    
    handler = ChangeHandler(mock_renderer)

    mock_event = MagicMock()
    mock_event.src_path = mock_renderer.script_path

    handler.on_modified(mock_event)
    assert mock_renderer.reload_pending is True

    mock_renderer.reload_pending = False
    other_event = MagicMock()
    other_event.src_path = os.path.abspath("other_file.py")
    handler.on_modified(other_event)
    assert mock_renderer.reload_pending is False

@patch('sdforge.render.NativeRenderer._init_window')
def test_render_with_debug_normals(mock_init):
    s = sphere()
    renderer = NativeRenderer(s, debug='normals')
    
    with patch('moderngl.create_context') as mock_create_context:
        mock_ctx = MagicMock()
        mock_create_context.return_value = mock_ctx
        renderer.ctx = mock_ctx

        renderer._compile_shader()

        call_args, call_kwargs = mock_ctx.program.call_args
        fragment_shader = call_kwargs.get('fragment_shader', '')
        
        assert "color = debugNormals(normal);" in fragment_shader
        assert "color = material_color * diffuse * ao;" not in fragment_shader

@patch('sdforge.render.NativeRenderer._init_window')
def test_render_with_debug_steps(mock_init):
    s = sphere()
    renderer = NativeRenderer(s, debug='steps')
    
    with patch('moderngl.create_context') as mock_create_context:
        mock_ctx = MagicMock()
        mock_create_context.return_value = mock_ctx
        renderer.ctx = mock_ctx

        renderer._compile_shader()
        
        call_args, call_kwargs = mock_ctx.program.call_args
        fragment_shader = call_kwargs.get('fragment_shader', '')

        assert "color = debugSteps(hit.z, 100.0);" in fragment_shader
        assert "debugNormals" in fragment_shader # The import is still there

@patch('sdforge.render.NativeRenderer._init_window')
def test_render_with_invalid_debug_mode(mock_init):
    s = sphere()
    renderer = NativeRenderer(s, debug='invalid_mode')
    with patch('moderngl.create_context') as mock_create_context:
        mock_ctx = MagicMock()
        mock_create_context.return_value = mock_ctx
        renderer.ctx = mock_ctx
        
        renderer._compile_shader()

        call_args, call_kwargs = mock_ctx.program.call_args
        fragment_shader = call_kwargs.get('fragment_shader', '')
        
        # Should fall back to standard lighting
        assert "color = material_color * diffuse * ao;" in fragment_shader
        assert "debugNormals" not in fragment_shader # Import should be skipped