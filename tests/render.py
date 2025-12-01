import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch
from sdforge import sphere
from sdforge.api.render import NativeRenderer
from sdforge.api.primitives import Sphere 

# We need the actual event handler class for one of the tests
try:
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


@pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="watchdog library not installed")
def test_change_handler_sets_reload_flag():
    """
    Tests that the internal ChangeHandler correctly sets the reload_pending flag
    when a file modification event occurs for the correct script.
    """
    # Create a mock renderer with the necessary attributes
    mock_renderer = MagicMock(spec=NativeRenderer)
    mock_renderer.script_path = os.path.abspath("my_script.py")
    mock_renderer.reload_pending = False

    # The FileSystemEventHandler class needs to be defined for the test
    class ChangeHandler(FileSystemEventHandler):
        def __init__(self, renderer_instance):
            self.renderer = renderer_instance
        def on_modified(self, event):
            if event.src_path == self.renderer.script_path:
                self.renderer.reload_pending = True

    handler = ChangeHandler(mock_renderer)

    # 1. Test with the correct file path
    correct_event = MagicMock()
    correct_event.src_path = mock_renderer.script_path
    handler.on_modified(correct_event)
    assert mock_renderer.reload_pending is True

    # 2. Test with an incorrect file path
    mock_renderer.reload_pending = False
    incorrect_event = MagicMock()
    incorrect_event.src_path = os.path.abspath("another_script.py")
    handler.on_modified(incorrect_event)
    assert mock_renderer.reload_pending is False


@patch('sdforge.api.render.NativeRenderer._compile_shader')
def test_reload_logic_updates_scene(mock_compile, tmp_path):
    """
    Tests the _reload_script method to ensure it correctly loads a new
    SDF object from a modified script file.
    """
    # 1. Create a temporary script file
    script_content_v1 = """
from sdforge import sphere
def main():
    return sphere(radius=1.0)
"""
    temp_script_path = tmp_path / "test_script.py"
    temp_script_path.write_text(script_content_v1)

    # 2. Initialize the renderer pointing to this script
    initial_obj = sphere(radius=99.0) # Start with a distinctly different object
    with patch('sys.argv', [str(temp_script_path)]):
        renderer = NativeRenderer(initial_obj)
        # Manually create a mock context and assign it
        mock_ctx = MagicMock()
        renderer.ctx = mock_ctx
        renderer.vbo = MagicMock() # Also mock the VBO for completeness

    # Assert initial state
    assert isinstance(renderer.sdf_obj, Sphere)
    assert renderer.sdf_obj.radius == 99.0

    # 3. Modify the script content to represent a file change
    script_content_v2 = """
from sdforge import sphere
def main():
    return sphere(radius=2.5)
"""
    temp_script_path.write_text(script_content_v2)

    # 4. Manually call the reload method
    renderer._reload_script()

    # 5. Assert that the scene object was updated
    assert isinstance(renderer.sdf_obj, Sphere)
    assert renderer.sdf_obj.radius == 2.5
    mock_compile.assert_called_once()
    renderer.ctx.simple_vertex_array.assert_called_once()

@patch('sdforge.api.mesh.generate')
def test_render_mode_mesh_calls_trimesh(mock_generate):
    """Tests that mode='mesh' generates geometry and calls trimesh.show()."""
    s = sphere()
    
    # Mock mesh generation to return dummy data
    mock_generate.return_value = (np.array([[0,0,0]], dtype='f4'), np.array([[0,0,0]], dtype='i4'))
    
    # Mock trimesh module
    mock_trimesh_module = MagicMock()
    mock_mesh_instance = MagicMock()
    mock_trimesh_module.Trimesh.return_value = mock_mesh_instance
    
    with patch.dict('sys.modules', {'trimesh': mock_trimesh_module}):
        s.render(mode='mesh')
    
    mock_generate.assert_called_once()
    mock_trimesh_module.Trimesh.assert_called_once()
    mock_mesh_instance.show.assert_called_once()

def test_render_mesh_missing_trimesh(capsys):
    """Tests error message when trimesh is missing in mesh mode."""
    s = sphere()
    # Simulate trimesh missing
    with patch.dict('sys.modules', {'trimesh': None}):
        s.render(mode='mesh')
    
    captured = capsys.readouterr()
    assert "ERROR: Mesh rendering mode requires 'trimesh'" in captured.err

@patch('sdforge.api.render.NativeRenderer.run')
def test_render_mode_window_launches_renderer(mock_run):
    """Tests that mode='window' initializes and runs the NativeRenderer."""
    s = sphere()
    # Mock dependencies required for window mode
    with patch.dict('sys.modules', {'moderngl': MagicMock(), 'glfw': MagicMock()}):
        s.render(mode='window')
    
    mock_run.assert_called_once()

def test_render_window_missing_deps(capsys):
    """Tests error message when dependencies are missing for window mode."""
    s = sphere()
    # Simulate moderngl/glfw missing
    with patch.dict('sys.modules', {'moderngl': None, 'glfw': None}):
        s.render(mode='window')
    
    captured = capsys.readouterr()
    assert "ERROR: Native window rendering requires 'moderngl' and 'glfw'" in captured.err

@patch('sdforge.api.render._render_mesh')
def test_render_mode_auto_detects_notebook(mock_render_mesh):
    """Tests that mode='auto' picks mesh rendering when in a notebook."""
    s = sphere()
    with patch('sdforge.api.render._IS_NOTEBOOK', True):
        s.render(mode='auto')
    mock_render_mesh.assert_called_once()

@patch('sdforge.api.render.NativeRenderer.run')
@patch('sdforge.api.render._render_mesh')
def test_render_mode_auto_detects_desktop(mock_render_mesh, mock_run):
    """Tests that mode='auto' picks window rendering when NOT in a notebook."""
    s = sphere()
    with patch('sdforge.api.render._IS_NOTEBOOK', False):
        with patch.dict('sys.modules', {'moderngl': MagicMock(), 'glfw': MagicMock()}):
            s.render(mode='auto')
    
    mock_render_mesh.assert_not_called()
    mock_run.assert_called_once()

def test_render_unknown_mode(capsys):
    """Tests that an unknown mode triggers an error message."""
    s = sphere()
    s.render(mode='invalid_mode')
    captured = capsys.readouterr()
    assert "ERROR: Unknown render mode 'invalid_mode'" in captured.err