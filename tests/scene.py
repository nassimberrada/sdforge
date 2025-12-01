import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch
from sdforge import sphere
from sdforge.api.scene import Camera, Light, NativeRenderer, Scene
from sdforge.api.primitives import Primitive 

try:
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

def test_camera_instantiation_defaults():
    cam = Camera()
    assert np.allclose(cam.position, (5, 4, 5))
    assert np.allclose(cam.target, (0, 0, 0))
    assert cam.zoom == 1.0

def test_camera_instantiation_custom_values():
    pos = (10, 20, 30)
    tgt = (1, 2, 3)
    zoom = 2.5
    cam = Camera(position=pos, target=tgt, zoom=zoom)
    assert np.allclose(cam.position, pos)
    assert np.allclose(cam.target, tgt)
    assert cam.zoom == zoom

def test_light_instantiation():
    light_static = Light(position=(5, 5, 5), ambient_strength=0.2, shadow_softness=16.0, ao_strength=5.0)
    assert np.allclose(light_static.position, (5, 5, 5))
    assert light_static.ambient_strength == 0.2
    assert light_static.shadow_softness == 16.0
    assert light_static.ao_strength == 5.0

def test_scene_class_init():
    s = sphere()
    cam = Camera()
    scene = Scene(s, cam)
    assert scene.root == s
    assert scene.camera == cam
    assert isinstance(scene.light, Light)

@pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="watchdog library not installed")
def test_change_handler_sets_reload_flag():
    # Setup mock renderer with path
    scene = Scene(sphere())
    mock_renderer = MagicMock(spec=NativeRenderer)
    mock_renderer.script_path = os.path.abspath("my_script.py")
    mock_renderer.reload_pending = False

    class ChangeHandler(FileSystemEventHandler):
        def __init__(self, renderer_instance): self.renderer = renderer_instance
        def on_modified(self, event):
            if event.src_path == self.renderer.script_path: self.renderer.reload_pending = True

    handler = ChangeHandler(mock_renderer)
    correct_event = MagicMock()
    correct_event.src_path = mock_renderer.script_path
    handler.on_modified(correct_event)
    assert mock_renderer.reload_pending is True

@patch('sdforge.api.scene.NativeRenderer._compile_shader')
@patch('sdforge.api.scene.NativeRenderer._init_context')
def test_reload_logic_updates_scene(mock_init, mock_compile, tmp_path):
    script_content_v1 = "from sdforge import sphere\ndef main(): return sphere(radius=1.0)"
    temp_script_path = tmp_path / "test_script.py"
    temp_script_path.write_text(script_content_v1)

    initial_scene = Scene(sphere(radius=99.0))
    
    with patch('sys.argv', [str(temp_script_path)]):
        renderer = NativeRenderer(initial_scene)
        renderer.ctx = MagicMock()
        renderer.vbo = MagicMock()
    
    # Pre-reload check
    assert isinstance(renderer.scene.root, Primitive)
    assert renderer.scene.root.radius == 99.0

    # Simulate edit
    script_content_v2 = "from sdforge import sphere\ndef main(): return sphere(radius=2.5)"
    temp_script_path.write_text(script_content_v2)

    # Reload
    renderer._reload_script()

    # Post-reload check
    assert isinstance(renderer.scene.root, Primitive)
    assert renderer.scene.root.radius == 2.5
    mock_compile.assert_called()

@patch('sdforge.api.scene.generate')
def test_render_mode_mesh_calls_trimesh(mock_generate):
    s = Scene(sphere())
    mock_generate.return_value = (np.array([[0,0,0]], dtype='f4'), np.array([[0,0,0]], dtype='i4'))
    
    mock_trimesh_module = MagicMock()
    mock_mesh_instance = MagicMock()
    mock_trimesh_module.Trimesh.return_value = mock_mesh_instance
    
    with patch.dict('sys.modules', {'trimesh': mock_trimesh_module}):
        s.render(mode='mesh')
        
    mock_generate.assert_called_once()
    mock_trimesh_module.Trimesh.assert_called_once()
    mock_mesh_instance.show.assert_called_once()

@patch('sdforge.api.scene.NativeRenderer.run')
def test_render_mode_window_launches_renderer(mock_run):
    s = Scene(sphere())
    with patch.dict('sys.modules', {'moderngl': MagicMock(), 'glfw': MagicMock()}):
        s.render(mode='window')
    mock_run.assert_called_once()

def test_render_window_missing_deps(capsys):
    s = Scene(sphere())
    with patch.dict('sys.modules', {'moderngl': None, 'glfw': None}):
        # NativeRenderer execution catches errors and prints them
        s.render(mode='window')
            
    captured = capsys.readouterr()
    # Check that the error was caught and logged
    assert "ERROR: Failed to launch window" in captured.err

@patch('sdforge.api.scene.Scene._render_mesh')
def test_render_mode_auto_detects_notebook(mock_render_mesh):
    s = Scene(sphere())
    with patch('sdforge.api.scene._IS_NOTEBOOK', True):
        s.render(mode='auto')
    mock_render_mesh.assert_called_once()

def test_render_unknown_mode(capsys):
    s = Scene(sphere())
    s.render(mode='invalid_mode')
    captured = capsys.readouterr()
    assert "ERROR: Unknown render mode 'invalid_mode'" in captured.err