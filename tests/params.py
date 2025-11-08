import pytest
from sdforge.api.params import Param

def test_param_init():
    p = Param("Size", 1.0, 0.0, 2.0)
    assert p.name == "Size"
    assert p.value == 1.0
    assert p.min_val == 0.0
    assert p.max_val == 2.0
    assert p.uniform_name.startswith("u_param_Size_")

def test_param_to_glsl():
    p = Param("My Param", 0.5, 0, 1)
    assert p.to_glsl() == p.uniform_name
    assert str(p) == p.uniform_name

def test_param_uniform_name_is_unique():
    p1 = Param("Same Name", 1, 0, 2)
    p2 = Param("Same Name", 1, 0, 2)
    assert p1.uniform_name != p2.uniform_name

def test_param_name_sanitization():
    p = Param("Invalid Name!@#$", 1, 0, 2)
    assert "Invalid" in p.uniform_name
    assert "!" not in p.uniform_name
    assert "@" not in p.uniform_name
    assert p.uniform_name.startswith("u_param_Invalid_Name____")