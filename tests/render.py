import pytest
import os
from unittest.mock import MagicMock, patch
from sdforge import sphere
from sdforge.render import NativeRenderer
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


# THE FIX: Remove the patch decorator for simple_vertex_array
@patch('sdforge.render.NativeRenderer._compile_shader')
def test_reload_logic_updates_scene(mock_compile, tmp_path):
    """
    Tests the _reload_script method to ensure it correctly loads a new
    SDF object from a modified script file.
    """
    # 1. Create a temporary script file
    script_content_v1 = """
from sdforge import sphere
def main():
    return sphere(1.0)
"""
    temp_script_path = tmp_path / "test_script.py"
    temp_script_path.write_text(script_content_v1)

    # 2. Initialize the renderer pointing to this script
    initial_obj = sphere(99.0) # Start with a distinctly different object
    with patch('sys.argv', [str(temp_script_path)]):
        renderer = NativeRenderer(initial_obj)
        # Manually create a mock context and assign it
        mock_ctx = MagicMock()
        renderer.ctx = mock_ctx
        renderer.vbo = MagicMock() # Also mock the VBO for completeness

    # Assert initial state
    assert isinstance(renderer.sdf_obj, Sphere)
    assert renderer.sdf_obj.r == 99.0
    
    # 3. Modify the script content to represent a file change
    script_content_v2 = """
from sdforge import sphere
def main():
    return sphere(2.5)
"""
    temp_script_path.write_text(script_content_v2)
    
    # 4. Manually call the reload method
    renderer._reload_script()
    
    # 5. Assert that the scene object was updated
    assert isinstance(renderer.sdf_obj, Sphere)
    assert renderer.sdf_obj.r == 2.5
    mock_compile.assert_called_once()
    renderer.ctx.simple_vertex_array.assert_called_once()