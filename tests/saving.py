import sys
import pytest
import os
from unittest.mock import patch, MagicMock, call
from sdforge import sphere, box, Forge

def test_save_static_object(tmp_path):
    s = sphere(radius=1.0) & box(size=1.5)
    output_file = tmp_path / "test_model.stl"
    # Test auto-bounding
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

@patch('sdforge.api.mesh._write_glb')
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
    assert "WARNING: Algorithm 'dual_contouring_fake' is not supported." in captured.err

def test_save_with_vertex_colors_warns(tmp_path, capsys):
    s = sphere(radius=1.0)
    output_file = tmp_path / "test.glb"
    with patch('sdforge.api.mesh._write_glb'):
        s.save(str(output_file), samples=2**10, verbose=False, vertex_colors=True)
    captured = capsys.readouterr()
    assert "WARNING: vertex_colors=True is not yet implemented for GLB export." in captured.err

@patch('sdforge.api.core.SDFNode.render')
def test_save_frame_api(mock_render):
    s = sphere()
    s.save_frame('test.png', camera=None, light=None)
    mock_render.assert_called_once_with(
        save_frame='test.png', 
        watch=False,
        camera=None,
        light=None,
    )

def test_save_displaced_object_fails(tmp_path):
    s = sphere(radius=1.0)
    displaced_sphere = s.displace("sin(p.x * 20.0) * 0.1")
    output_file = tmp_path / "displaced.stl"
    with pytest.raises(TypeError, match="Cannot create a callable for an object with raw GLSL displacement"):
        displaced_sphere.save(str(output_file), verbose=False)

def test_save_unsupported_format(tmp_path, capsys):
    s = sphere(radius=1.0)
    output_file = tmp_path / "test_model.ply"
    s.save(str(output_file), verbose=False)
    captured = capsys.readouterr()
    assert "ERROR: Unsupported file format" in captured.err

def test_save_marching_cubes_failure(tmp_path, capsys):
    s = sphere(radius=0.1).translate((10, 10, 10))
    output_file = tmp_path / "no_intersect.stl"
    s.save(str(output_file), bounds=((-1, -1, -1), (1, 1, 1)), samples=2**10, verbose=False)
    captured = capsys.readouterr()
    assert "ERROR: Mesh generation failed" in captured.err

@pytest.mark.skipif("trimesh" not in sys.modules, reason="trimesh library not installed")
def test_save_with_decimation_simplifies_mesh(tmp_path):
    """Tests that decimation actually reduces the file size."""
    s = sphere(radius=1.0)

    original_file = tmp_path / "original.stl"
    decimated_file = tmp_path / "decimated.stl"

    s.save(str(original_file), samples=2**16, verbose=False)
    s.save(str(decimated_file), samples=2**16, verbose=False, decimate_ratio=0.9)

    assert os.path.exists(original_file)
    assert os.path.exists(decimated_file)

    original_size = os.path.getsize(original_file)
    decimated_size = os.path.getsize(decimated_file)

    assert decimated_size > 84
    assert decimated_size < original_size

@patch.dict('sys.modules', {'trimesh': None})
def test_save_with_decimation_warns_if_trimesh_missing(tmp_path, capsys):
    """Tests that a warning is printed if decimation is requested but trimesh is missing."""
    s = sphere(radius=1.0)
    output_file = tmp_path / "test.stl"

    s.save(str(output_file), samples=2**10, verbose=False, decimate_ratio=0.5)

    captured = capsys.readouterr()
    assert "WARNING: Mesh simplification requires the 'trimesh' library." in captured.err
    assert "pip install trimesh" in captured.err

def test_save_with_invalid_decimation_ratio_warns(tmp_path, capsys):
    """Tests that an invalid ratio prints a warning and does not simplify."""
    s = sphere(radius=1.0)
    original_file = tmp_path / "original.stl"
    decimated_file = tmp_path / "decimated.stl"

    s.save(str(original_file), samples=2**12, verbose=False)
    s.save(str(decimated_file), samples=2**12, verbose=False, decimate_ratio=1.5) # Invalid ratio

    captured = capsys.readouterr()
    assert "WARNING: `decimate_ratio` must be between 0 and 1." in captured.err

    # The file sizes should be identical since simplification was skipped
    assert os.path.getsize(original_file) == os.path.getsize(decimated_file)

# --- Adaptive Meshing Tests ---

def test_save_adaptive_object(tmp_path):
    """Tests that adaptive meshing runs and creates a valid file."""
    s = sphere(radius=1.0)
    output_file = tmp_path / "adaptive_model.stl"
    s.save(str(output_file), adaptive=True, octree_depth=6, verbose=False)
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 84

def test_save_adaptive_with_samples_warns(tmp_path, capsys):
    """Tests that using both adaptive and samples flags gives a warning."""
    s = sphere(radius=1.0)
    output_file = tmp_path / "adaptive_warn.stl"
    s.save(str(output_file), adaptive=True, samples=2**10, verbose=False)
    captured = capsys.readouterr()
    assert "WARNING: `samples` parameter is ignored when `adaptive=True`" in captured.err

def test_adaptive_is_smarter_than_uniform_for_sparse_scene():
    """
    Tests the core benefit of adaptive meshing: fewer evaluations for sparse scenes.
    """
    # A sparse scene: a small sphere in a large bounding box
    s = sphere(radius=0.1)
    bounds = ((-5, -5, -5), (5, 5, 5))

    # Mock the SDF's callable to count how many times it's invoked
    mock_callable = MagicMock(wraps=s.to_callable())
    s.to_callable = MagicMock(return_value=mock_callable)

    # Run uniform meshing
    # We patch the file writer to avoid actual disk I/O
    with patch('sdforge.api.mesh._write_binary_stl'):
        s.save("uniform.stl", bounds=bounds, adaptive=False, samples=10**3, verbose=False)
    uniform_calls = mock_callable.call_args[0][0].shape[0]

    mock_callable.reset_mock()

    # Run adaptive meshing
    with patch('sdforge.api.mesh._write_binary_stl'):
        s.save("adaptive.stl", bounds=bounds, adaptive=True, octree_depth=5, verbose=False)
    adaptive_calls = mock_callable.call_args[0][0].shape[0]

    # The number of uniform calls should be around 1000 (10x10x10 grid).
    # The number of adaptive calls should be significantly less, as it only
    # explores the area immediately around the small sphere.
    assert uniform_calls > 900
    assert adaptive_calls < 500
    assert adaptive_calls < uniform_calls

def test_save_dual_contouring(tmp_path):
    """Tests that dual contouring runs and creates a valid file."""
    s = box(size=1.1)
    output_file = tmp_path / "dc_model.stl"
    s.save(str(output_file), samples=2**12, verbose=False, algorithm='dual_contouring')
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 84

def test_save_adaptive_dual_contouring_fails(tmp_path):
    """Tests that trying to use adaptive meshing with dual contouring raises an error."""
    s = sphere(radius=1.0)
    output_file = tmp_path / "test.stl"
    with pytest.raises(ValueError, match="does not currently support adaptive meshing"):
        s.save(str(output_file), adaptive=True, algorithm='dual_contouring', verbose=False)

def test_save_with_voxel_size(tmp_path):
    """
    Tests that specifying voxel_size generates a mesh file.
    """
    s = sphere(radius=1.0)
    output_file = tmp_path / "voxel_model.stl"
    
    # Use a coarse voxel size for speed
    s.save(str(output_file), voxel_size=0.5, verbose=False)
    
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 84

def test_save_with_voxel_size_and_adaptive(tmp_path, capsys):
    """
    Tests that voxel_size works with adaptive meshing and logs the calculated depth.
    """
    s = box(size=4.0)
    output_file = tmp_path / "voxel_adaptive.stl"
    
    # 4.0 size / 0.5 voxel = 8 steps -> 2^3 = 8 -> depth should be around 3 or 4
    # Max dim might be slightly padded.
    s.save(str(output_file), voxel_size=0.5, adaptive=True, verbose=True)
    
    captured = capsys.readouterr()
    assert "implies octree_depth=" in captured.out
    assert os.path.exists(output_file)