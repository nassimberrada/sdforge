import pytest
from sdforge.api.primitives.function import Function
from sdforge.api.core import GLSLContext

class MockCompiler:
    pass

def test_function_initialization():
    """Tests that Function node initializes correctly with/without return statements."""
    # Without return statement
    node1 = Function("p.x*p.x + p.y*p.y - 1.0", safety=0.25)
    assert "return p.x*p.x" in node1.glsl_code_body
    assert node1.safety == 0.25

    # With return statement
    node2 = Function("return sin(p.x);")
    assert node2.glsl_code_body == "return sin(p.x);"
    assert node2.safety == 0.5 # Default safety

def test_function_uniforms():
    """Tests that custom uniforms are correctly stored and collected."""
    node = Function("p.x - u_offset", uniforms={'u_offset': 5.0})
    
    collected = {}
    node._collect_uniforms(collected)
    assert 'u_offset' in collected
    assert collected['u_offset'] == 5.0

def test_function_glsl_generation():
    """Tests that the complex numerical gradient GLSL is correctly generated."""
    node = Function("p.x*p.x - 1.0", safety=0.5)
    ctx = GLSLContext(MockCompiler())
    
    result_var = node.to_glsl(ctx)
    
    # Check that the global definition was added
    assert any("implicit_func_" in defn for defn in ctx.definitions)
    assert any("return p.x*p.x - 1.0;" in defn for defn in ctx.definitions)
    
    statements = "\n".join(ctx.statements)
    
    # Check for numerical gradient evaluation swizzles
    assert ".xyy" in statements
    assert ".yxy" in statements
    assert ".yyx" in statements
    
    # Check for division by gradient length + epsilon
    assert "length(" in statements
    assert "0.0001" in statements
    
    # Check for safety factor and clamping
    assert "0.5" in statements
    assert "clamp(" in statements
    assert "-1.0, 1.0" in statements

def test_function_cpu_backend_fails():
    """Tests that the CPU backend gracefully rejects Function nodes."""
    node = Function("p.x - 1.0")
    with pytest.raises(NotImplementedError) as excinfo:
        node.to_callable(backend='cpu')
    assert "not supported by the CPU backend" in str(excinfo.value)