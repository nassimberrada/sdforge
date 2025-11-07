import pytest
import os
from sdforge import sphere, box, Param

def test_export_shader_creates_file(tmp_path):
    s = sphere(1.0) & box(1.5).color(1, 0, 0)
    output_file = tmp_path / "test_shader.glsl"
    
    s.export_shader(str(output_file))
    
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
        assert "void main()" in content
        assert "vec4 Scene(in vec3 p)" in content
        assert "sdSphere" in content
        assert "sdBox" in content
        assert "opI" in content
        assert "struct MaterialInfo" in content
        assert "uniform MaterialInfo u_materials[1];" in content

def test_export_shader_with_param(tmp_path):
    p_size = Param("Box Size", 1.5, 1.0, 2.0)
    s = box(size=p_size)
    output_file = tmp_path / "param_shader.glsl"

    s.export_shader(str(output_file))
    
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
        assert f"uniform float {p_size.uniform_name};" in content