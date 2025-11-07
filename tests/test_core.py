import pytest
import numpy as np
from sdforge import sphere, box, Group

def test_estimate_bounds_simple():
    s = sphere(1.0)
    bounds = s.estimate_bounds(resolution=32, verbose=False)
    min_b, max_b = bounds
    
    # Check if the bounds are reasonable for a unit sphere
    assert -1.2 < min_b[0] < -0.9
    assert -1.2 < min_b[1] < -0.9
    assert -1.2 < min_b[2] < -0.9
    assert 0.9 < max_b[0] < 1.2
    assert 0.9 < max_b[1] < 1.2
    assert 0.9 < max_b[2] < 1.2

def test_estimate_bounds_transformed():
    s = sphere(0.5).translate((5, 0, 0))
    # Need to expand search_bounds and use higher resolution for accuracy
    bounds = s.estimate_bounds(resolution=128, search_bounds=((-10,-10,-10), (10,10,10)), verbose=False)
    min_b, max_b = bounds
    
    # Check if bounds are centered around (5, 0, 0)
    assert 4.0 < min_b[0] < 4.6
    assert 5.4 < max_b[0] < 6.0

def test_estimate_bounds_no_object_found():
    s = sphere(0.5).translate((100, 100, 100))
    search_bounds = ((-2, -2, -2), (2, 2, 2))
    bounds = s.estimate_bounds(search_bounds=search_bounds, verbose=False)
    
    # Should return the original search_bounds
    assert bounds == search_bounds

def test_estimate_bounds_animated_fails():
    s = sphere(r="1.0 + sin(u_time)")
    with pytest.raises(TypeError):
        s.estimate_bounds(verbose=False)

def test_estimate_bounds_with_group():
    s1 = sphere(0.5).translate((-2, 0, 0))
    s2 = sphere(0.5).translate((2, 0, 0))
    g = Group(s1, s2)
    
    bounds = g.estimate_bounds(resolution=64, verbose=False)
    min_b, max_b = bounds

    # Check if bounds encompass both spheres
    assert -2.7 < min_b[0] < -2.4
    assert 2.4 < max_b[0] < 2.7
    assert -0.7 < min_b[1] < -0.4
    assert 0.4 < max_b[1] < 0.7