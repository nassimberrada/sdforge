import pytest
import numpy as np
from sdforge import Sketch, SDFNode
from sdforge.api.operations import Union
from sdforge.render import SceneCompiler
from tests.conftest import requires_glsl_validator

def test_sketch_init():
    s = Sketch(start=(1, 2))
    assert np.allclose(s._current_pos, [1, 2])
    assert np.allclose(s._start_pos, [1, 2])
    assert len(s._segments) == 0

def test_sketch_chaining():
    """Tests that methods return self for chaining."""
    s = Sketch()
    res = s.line_to(1, 0).curve_to(2, 1, (1.5, 0)).close()
    assert res is s
    assert len(s._segments) == 3

def test_sketch_segments_data():
    """Tests that geometric data is stored correctly in 3D format."""
    s = Sketch(start=(0, 0))
    s.line_to(1, 0)
    s.curve_to(1, 1, control=(2, 0))

    # Segment 0: Line
    seg0 = s._segments[0]
    assert seg0['type'] == 'line'
    assert np.allclose(seg0['start'], [0, 0, 0])
    assert np.allclose(seg0['end'], [1, 0, 0])

    # Segment 1: Bezier
    seg1 = s._segments[1]
    assert seg1['type'] == 'bezier'
    assert np.allclose(seg1['start'], [1, 0, 0])
    assert np.allclose(seg1['end'], [1, 1, 0])
    assert np.allclose(seg1['control'], [2, 0, 0])

def test_sketch_close():
    """Tests that close() adds a line back to start."""
    s = Sketch(start=(1, 1))
    s.line_to(2, 2)
    s.close()

    assert len(s._segments) == 2
    last_seg = s._segments[-1]
    assert last_seg['type'] == 'line'
    assert np.allclose(last_seg['start'], [2, 2, 0])
    assert np.allclose(last_seg['end'], [1, 1, 0])

def test_sketch_to_sdf():
    """Tests conversion to SDFNode."""
    s = Sketch().line_to(1,0).to_sdf(stroke_radius=0.1)
    assert isinstance(s, SDFNode)
    assert isinstance(s, Union)
    assert len(s.children) == 1

def test_sketch_callable():
    """Tests numerical evaluation of the sketch SDF."""
    # Horizontal line from (0,0) to (2,0)
    s = Sketch(start=(0,0)).line_to(2,0).to_sdf(stroke_radius=0.1)

    callable_fn = s.to_callable()

    points = np.array([
        [1.0, 0.0, 0.0],  # Midpoint on line, dist = -radius (-0.1)
        [1.0, 0.2, 0.0],  # 0.2 units away from center, dist = 0.2 - 0.1 = 0.1
        [3.0, 0.0, 0.0]   # 1.0 unit past end, dist = 1.0 - 0.1 = 0.9
    ])

    dists = callable_fn(points)

    assert np.isclose(dists[0], -0.1, atol=1e-5)
    assert np.isclose(dists[1], 0.1, atol=1e-5)
    assert np.isclose(dists[2], 0.9, atol=1e-5)

@requires_glsl_validator
def test_sketch_glsl_compiles(validate_glsl):
    """Tests that the generated GLSL for a sketch is valid."""
    s = Sketch(start=(0,0)).line_to(1,0).curve_to(1,1, (0.5, 0.5)).close().to_sdf()
    scene_code = SceneCompiler().compile(s)
    validate_glsl(scene_code)

def test_sketch_profile_logic():
    """
    Tests that the sketch correctly acts as a 2D profile (ignoring Z in 2D mode).
    If we ask for `to_profile_callable`, it should evaluate based on x,y even if z is huge.
    """
    sketch = Sketch().line_to(1,0).to_sdf(stroke_radius=0.1)
    prof_callable = sketch.to_profile_callable()

    # Point at (0.5, 0) but with huge Z
    # Since it's a profile callable, it should project Z to 0 and find the distance to the line
    p = np.array([[0.5, 0.0, 100.0]]) 
    d = prof_callable(p)

    assert np.isclose(d[0], -0.1, atol=1e-5)