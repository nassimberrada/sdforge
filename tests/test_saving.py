import pytest
from sdforge import *
import os
import sys
from unittest.mock import MagicMock

def test_save_static_object(tmp_path):
    s = sphere(1.0) & box(1.5)
    output_file = tmp_path / "test_model.stl"
    s.save(str(output_file), samples=2**12, verbose=False)
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 84

def test_save_animated_object_fails(tmp_path):
    animated_sphere = sphere(r="0.5 + 0.2 * sin(u_time)")
    output_file = tmp_path / "animated.stl"
    with pytest.raises(TypeError, match="Cannot save mesh of an object with animated"):
        animated_sphere.save(str(output_file), verbose=False)

def test_to_callable_animated_object_fails():
    animated_sphere = sphere(r="0.5 + 0.2 * sin(u_time)")
    with pytest.raises(TypeError, match="Cannot save mesh of an object with animated"):
        animated_sphere.to_callable()

def test_to_callable_displaced_object_fails():
    s = sphere(1.0)
    displaced_sphere = s.displace("sin(p.x * 20.0) * 0.1")
    with pytest.raises(TypeError, match="Cannot save mesh of an object with GLSL-based displacement"):
        displaced_sphere.to_callable()

def test_to_callable_forge_object_requires_deps(monkeypatch):
    monkeypatch.setattr('sdforge.api._MODERNGL_AVAILABLE', False)
    f = Forge("return length(p) - 1.0;")
    with pytest.raises(ImportError, match="To save meshes with Forge objects"):
        f.to_callable()

def test_save_unsupported_format(tmp_path, capsys):
    s = sphere(1.0)
    output_file = tmp_path / "test_model.obj"
    s.save(str(output_file), verbose=False)
    captured = capsys.readouterr()
    assert "ERROR: Unsupported file format" in captured.err or "ERROR: Unsupported file format" in captured.out

def test_save_marching_cubes_failure(tmp_path, capsys):
    s = sphere(0.1).translate((10, 10, 10))
    output_file = tmp_path / "no_intersect.stl"
    s.save(str(output_file), samples=2**10, verbose=False)
    captured = capsys.readouterr()
    assert "ERROR: Marching cubes failed" in captured.err or "ERROR: Marching cubes failed" in captured.out

def test_to_callable_forge_object_succeeds_with_deps(monkeypatch):
    monkeypatch.delattr('sdforge.api.Forge._mgl_context', raising=False)
    monkeypatch.setattr('sdforge.api._MODERNGL_AVAILABLE', True)
    
    mock_glfw = MagicMock()
    mock_mgl = MagicMock()
    
    mock_mgl.create_context.return_value = MagicMock()

    monkeypatch.setattr('sdforge.api.glfw', mock_glfw)
    monkeypatch.setattr('sdforge.api.moderngl', mock_mgl)
    
    f = Forge("return length(p) - 1.0;")
    
    try:
        callable_func = f.to_callable()
        assert callable(callable_func)
    except Exception as e:
        pytest.fail(f"Forge.to_callable() failed with dependencies mocked: {e}")
    
    mock_glfw.init.assert_called()
    mock_mgl.create_context.assert_called()