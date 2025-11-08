import pytest
import os
from unittest.mock import patch
from sdforge import sphere, box, Forge

def test_save_static_object(tmp_path):
    s = sphere(1.0) & box(1.5)
    output_file = tmp_path / "test_model.stl"
    # Test auto-bounding
    s.save(str(output_file), samples=2**12, verbose=False)
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 84

def test_save_obj_static_object(tmp_path):
    s = sphere(1.0) & box(1.5)
    output_file = tmp_path / "test_model.obj"
    s.save(str(output_file), samples=2**12, verbose=False)
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
        assert 'v ' in content
        assert 'f ' in content

@patch('sdforge.mesh._write_glb')
def test_save_glb_calls_writer(mock_write_glb, tmp_path):
    s = sphere(1.0)
    output_file = tmp_path / "test_model.glb"
    s.save(str(output_file), samples=2**10, verbose=False)
    mock_write_glb.assert_called_once()

def test_save_with_advanced_meshing_warns(tmp_path, capsys):
    s = sphere(1.0)
    output_file = tmp_path / "test.stl"
    s.save(str(output_file), samples=2**10, verbose=False, algorithm='dual_contouring', adaptive=True)
    captured = capsys.readouterr()
    assert "WARNING: Algorithm 'dual_contouring' is not supported." in captured.err
    assert "WARNING: Adaptive meshing is not yet implemented." in captured.err

def test_save_with_vertex_colors_warns(tmp_path, capsys):
    s = sphere(1.0)
    output_file = tmp_path / "test.glb"
    with patch('sdforge.mesh._write_glb'):
        s.save(str(output_file), samples=2**10, verbose=False, vertex_colors=True)
    captured = capsys.readouterr()
    assert "WARNING: vertex_colors=True is not yet implemented for GLB export." in captured.err

@patch('sdforge.core.SDFNode.render')
def test_save_frame_api(mock_render):
    s = sphere()
    s.save_frame('test.png', time=1.23)
    mock_render.assert_called_once_with(
        save_frame='test.png', 
        watch=False,
        camera=None,
        light=None,
        time=1.23
    )

def test_save_displaced_object_fails(tmp_path):
    s = sphere(1.0)
    displaced_sphere = s.displace("sin(p.x * 20.0) * 0.1")
    output_file = tmp_path / "displaced.stl"
    with pytest.raises(TypeError, match="Cannot create a callable for an object with raw GLSL displacement"):
        displaced_sphere.save(str(output_file), verbose=False)

def test_save_unsupported_format(tmp_path, capsys):
    s = sphere(1.0)
    output_file = tmp_path / "test_model.ply"
    s.save(str(output_file), verbose=False)
    captured = capsys.readouterr()
    assert "ERROR: Unsupported file format" in captured.err

def test_save_marching_cubes_failure(tmp_path, capsys):
    s = sphere(0.1).translate((10, 10, 10))
    output_file = tmp_path / "no_intersect.stl"
    s.save(str(output_file), bounds=((-1, -1, -1), (1, 1, 1)), samples=2**10, verbose=False)
    captured = capsys.readouterr()
    assert "ERROR: Marching cubes failed" in captured.err