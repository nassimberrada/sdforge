import pytest
import os
from sdforge import sphere, box, Param, Forge

def test_export_shader_creates_file(tmp_path):
    """Tests that a file is created and has the basic shader structure."""
    s = sphere(1.0) & box(1.5)
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
    """Tests that GLSL code for used nodes is included in the export."""
    s = sphere(1.0).twist(2.0)
    output_file = tmp_path / "deps_shader.glsl"
    
    s.export_shader(str(output_file))

    with open(output_file, 'r') as f:
        content = f.read()
        # Check for function names from different library files
        assert "sdSphere" in content # from primitives.glsl
        assert "opTwist" in content  # from transforms.glsl

def test_export_shader_with_param_and_forge(tmp_path):
    """Tests that uniforms from both Param and Forge objects are declared."""
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