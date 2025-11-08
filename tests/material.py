import pytest
from sdforge.api.material import Material
from sdforge import sphere, box
from sdforge.core import GLSLContext, SDFNode

def test_material_api():
    """Tests that the .color() method creates a Material wrapper."""
    s = sphere(1.0)
    colored_s = s.color(1.0, 0.5, 0.2)
    
    assert isinstance(colored_s, Material)
    assert colored_s.child == s
    assert colored_s.color == (1.0, 0.5, 0.2)

def test_collect_materials():
    """Tests the recursive collection of unique materials."""
    s1 = sphere(1.0).color(1, 0, 0)
    s2 = sphere(0.5).color(0, 1, 0)
    b1 = box(1.0).color(1, 0, 0) # Same material as s1
    
    scene = (s1 | s2) - b1
    
    materials = []
    scene._collect_materials(materials)
    
    # Should find 2 unique materials (red and green)
    assert len(materials) == 2
    
    # Check that IDs were assigned correctly
    assert s1.material_id == 0
    assert s2.material_id == 1
    # b1 shares a material object with s1, so it should have the same ID
    assert b1.material_id == 0
    
    assert materials[0].color == (1, 0, 0)
    assert materials[1].color == (0, 1, 0)

def test_material_glsl_generation():
    """Tests that the material ID is correctly injected into the GLSL."""
    s = sphere(1.0)
    mat = Material(s, (1, 0, 0))
    mat.material_id = 5 # Manually assign an ID for testing
    
    ctx = GLSLContext(compiler=None)
    result_var = mat.to_glsl(ctx)
    
    # Check that the statements create a new vec4 with the correct material ID
    assert f"vec4 {result_var} = vec4(var_0.x, 5.0, var_0.zw);" in "\n".join(ctx.statements)