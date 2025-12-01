import sys
import pytest
import os
from unittest.mock import patch, MagicMock
from sdforge import sphere, box, Param, Forge

def test_export_shader_creates_file(tmp_path):
    s = sphere(radius=1.0) & box(size=1.5)
    output_file = tmp_path / "test_shader.glsl"
    s.export_shader(str(output_file))
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
        assert "void main()" in content
        assert "vec4 Scene(in vec3 p)" in content
        assert "raymarch" in content
        assert "cameraStatic" in content

def test_export_shader_includes_dependencies(tmp_path):
    s = sphere(radius=1.0).twist(2.0)
    output_file = tmp_path / "deps_shader.glsl"
    s.export_shader(str(output_file))
    with open(output_file, 'r') as f:
        content = f.read()
        assert "sdSphere" in content 
        assert "opTwist" in content

def test_export_shader_with_param_and_forge(tmp_path):
    p_size = Param("Box Size", 1.5, 1.0, 2.0)
    f = Forge("length(p) - u_radius", uniforms={'u_radius': 0.5})
    scene = box(size=p_size) | f
    output_file = tmp_path / "param_shader.glsl"
    scene.export_shader(str(output_file))
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
        assert f"uniform float {p_size.uniform_name};" in content
        assert "uniform float u_radius;" in content

def test_save_static_object(tmp_path):
    s = sphere(radius=1.0) & box(size=1.5)
    output_file = tmp_path / "test_model.stl"
    s.save(str(output_file), samples=2**12, verbose=False)
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 84

def test_save_obj_static_object(tmp_path):
    s = sphere(radius=1.0) & box(size=1.5)
    output_file = tmp_path / "test_model.obj"
    s.save(str(output_file), samples=2**12, verbose=False)
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
        assert 'v ' in content
        assert 'f ' in content

@patch('sdforge.api.io._write_glb')
def test_save_glb_calls_writer(mock_write_glb, tmp_path):
    s = sphere(radius=1.0)
    output_file = tmp_path / "test_model.glb"
    s.save(str(output_file), samples=2**10, verbose=False)
    mock_write_glb.assert_called_once()

def test_save_with_unsupported_algorithm_warns(tmp_path, capsys):
    s = sphere(radius=1.0)
    output_file = tmp_path / "test.stl"
    s.save(str(output_file), samples=2**10, verbose=False, algorithm='dual_contouring_fake')
    captured = capsys.readouterr()
    assert "Algorithm 'dual_contouring_fake' is not supported." not in captured.err

def test_save_with_vertex_colors_warns(tmp_path, capsys):
    s = sphere(radius=1.0)
    output_file = tmp_path / "test.glb"
    with patch('sdforge.api.io._write_glb'):
        s.save(str(output_file), samples=2**10, verbose=False, vertex_colors=True)
    captured = capsys.readouterr()
    assert "WARNING: vertex_colors not implemented for GLB." in captured.err

@patch('sdforge.api.core.SDFNode.render')
def test_save_frame_api(mock_render):
    s = sphere()
    s.save_frame('test.png', camera=None, light=None)
    mock_render.assert_called_once_with(save_frame='test.png', watch=False, camera=None, light=None)

def test_save_displaced_object_fails(tmp_path):
    s = sphere(radius=1.0).displace("sin(p.x * 20.0) * 0.1")
    output_file = tmp_path / "displaced.stl"
    with pytest.raises(TypeError, match="GPU-only"):
        s.save(str(output_file), verbose=False)

def test_save_unsupported_format(tmp_path, capsys):
    s = sphere(radius=1.0)
    output_file = tmp_path / "test_model.ply"
    s.save(str(output_file), verbose=False)
    captured = capsys.readouterr()
    assert "ERROR: Unsupported format" in captured.err

def test_save_marching_cubes_failure(tmp_path, capsys):
    s = sphere(radius=0.1).translate((10, 10, 10))
    output_file = tmp_path / "no_intersect.stl"
    s.save(str(output_file), bounds=((-1, -1, -1), (1, 1, 1)), samples=2**10, verbose=False)
    captured = capsys.readouterr()
    assert "ERROR: Mesh generation failed" in captured.err

@pytest.mark.skipif("trimesh" not in sys.modules, reason="trimesh library not installed")
def test_save_with_decimation_simplifies_mesh(tmp_path):
    s = sphere(radius=1.0)
    original_file = tmp_path / "original.stl"
    decimated_file = tmp_path / "decimated.stl"
    s.save(str(original_file), samples=2**16, verbose=False)
    s.save(str(decimated_file), samples=2**16, verbose=False, decimate_ratio=0.9)
    assert os.path.exists(original_file)
    assert os.path.exists(decimated_file)
    assert os.path.getsize(decimated_file) < os.path.getsize(original_file)

@patch.dict('sys.modules', {'trimesh': None})
def test_save_with_decimation_warns_if_trimesh_missing(tmp_path, capsys):
    s = sphere(radius=1.0)
    output_file = tmp_path / "test.stl"
    s.save(str(output_file), samples=2**10, verbose=False, decimate_ratio=0.5)
    captured = capsys.readouterr()
    assert "WARNING: 'trimesh' required for simplification." in captured.err

def test_save_adaptive_object(tmp_path):
    s = sphere(radius=1.0)
    output_file = tmp_path / "adaptive_model.stl"
    s.save(str(output_file), adaptive=True, octree_depth=6, verbose=False)
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 84

def test_adaptive_is_smarter_than_uniform_for_sparse_scene():
    s = sphere(radius=0.1)
    bounds = ((-5, -5, -5), (5, 5, 5))
    mock_callable = MagicMock(wraps=s.to_callable())
    s.to_callable = MagicMock(return_value=mock_callable)
    with patch('sdforge.api.io._write_binary_stl'):
        s.save("uniform.stl", bounds=bounds, adaptive=False, samples=10**3, verbose=False)
    uniform_calls = mock_callable.call_args[0][0].shape[0]
    mock_callable.reset_mock()
    with patch('sdforge.api.io._write_binary_stl'):
        s.save("adaptive.stl", bounds=bounds, adaptive=True, octree_depth=5, verbose=False)
    adaptive_calls = mock_callable.call_args[0][0].shape[0]
    assert uniform_calls > 900
    assert adaptive_calls < 500
    assert adaptive_calls < uniform_calls

def test_save_dual_contouring(tmp_path):
    s = box(size=1.1)
    output_file = tmp_path / "dc_model.stl"
    s.save(str(output_file), samples=2**12, verbose=False, algorithm='dual_contouring')
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 84

def test_save_adaptive_dual_contouring_fails(tmp_path):
    s = sphere(radius=1.0)
    output_file = tmp_path / "test.stl"
    with pytest.raises(ValueError, match="Dual Contouring does not support adaptive meshing"):
        s.save(str(output_file), adaptive=True, algorithm='dual_contouring', verbose=False)

def test_save_with_voxel_size(tmp_path):
    s = sphere(radius=1.0)
    output_file = tmp_path / "voxel_model.stl"
    s.save(str(output_file), voxel_size=0.5, verbose=False)
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 84