import numpy as np
from sdforge import Sketch, SDFNode
from sdforge.api.operations import Union
from sdforge.api.render import SceneCompiler
from tests.conftest import requires_glsl_validator
from sdforge.api.primitives import Line, Bezier # Import for type checking

def test_sketch_init():
    s = Sketch(start=(1, 2))
    assert np.allclose(s._current_pos, [1, 2])
    assert np.allclose(s._start_pos, [1, 2])
    assert len(s._segments) == 0

def test_sketch_chaining():
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

def test_sketch_close_straight_line():
    """Tests that close() adds a straight line back to start by default."""
    s = Sketch(start=(1, 1))
    s.line_to(2, 2)
    s.close()

    assert len(s._segments) == 2
    last_seg = s._segments[-1]
    assert last_seg['type'] == 'line'
    assert np.allclose(last_seg['start'], [2, 2, 0])
    assert np.allclose(last_seg['end'], [1, 1, 0])

def test_sketch_close_curved_bezier():
    """Tests that close() adds a Bezier curve when curve_control is provided."""
    s = Sketch(start=(0, 0))
    s.line_to(1, 0)
    control_point = (0.5, 0.5)
    s.close(curve_control=control_point)

    assert len(s._segments) == 2
    last_seg = s._segments[-1]
    assert last_seg['type'] == 'bezier'
    assert np.allclose(last_seg['start'], [1, 0, 0])
    assert np.allclose(last_seg['control'], [*control_point, 0.0]) # Z-coord added by _to_vec3
    assert np.allclose(last_seg['end'], [0, 0, 0])
    
def test_sketch_to_sdf():
    s = Sketch().line_to(1,0).to_sdf(stroke_radius=0.1)
    assert isinstance(s, SDFNode)
    assert isinstance(s, Union)
    assert len(s.children) == 1
    assert isinstance(s.children[0], Line) # Should be a Line primitive

def test_sketch_to_sdf_curved_closure():
    s = Sketch(start=(0,0)).line_to(1,0).close(curve_control=(0.5, 0.5)).to_sdf(stroke_radius=0.1)
    assert isinstance(s, SDFNode)
    assert isinstance(s, Union)
    assert len(s.children) == 2
    assert isinstance(s.children[0], Line)
    assert isinstance(s.children[1], Bezier) # The closure should be a Bezier primitive

def test_sketch_callable():
    s = Sketch(start=(0,0)).line_to(2,0).to_sdf(stroke_radius=0.1)
    callable_fn = s.to_callable()
    points = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
        [3.0, 0.0, 0.0]
    ])
    dists = callable_fn(points)
    assert np.isclose(dists[0], -0.1, atol=1e-4)
    assert np.isclose(dists[1], 0.1, atol=1e-4)
    assert np.isclose(dists[2], 0.9, atol=1e-4)

@requires_glsl_validator
def test_sketch_glsl_compiles(validate_glsl):
    # Test with straight line closure
    s_line = Sketch(start=(0,0)).line_to(1,0).curve_to(1,1, (0.5, 0.5)).close().to_sdf()
    scene_code_line = SceneCompiler().compile(s_line)
    validate_glsl(scene_code_line)
    
    # Test with curved closure
    s_curve = Sketch(start=(0,0)).line_to(1,0).close(curve_control=(0.5, 0.5)).to_sdf()
    scene_code_curve = SceneCompiler().compile(s_curve)
    validate_glsl(scene_code_curve)

def test_sketch_profile_logic():
    sketch = Sketch().line_to(1,0).to_sdf(stroke_radius=0.1)
    prof_callable = sketch.to_profile_callable()
    
    # Point at (0.5, 0) but with huge Z. Profile should ignore Z.
    p = np.array([[0.5, 0.0, 100.0]]) 
    d = prof_callable(p)
    assert np.isclose(d[0], -0.1, atol=1e-4)