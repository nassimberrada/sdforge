import pytest
from sdforge import *
import numpy as np

@pytest.fixture
def shapes():
    s = sphere(1)
    b = box(1.5)
    return s, b

def test_union(shapes):
    s, b = shapes
    u = s | b
    assert isinstance(u, SDFObject)
    assert "opU(" in u.to_glsl()

def test_intersection(shapes):
    s, b = shapes
    i = s & b
    assert isinstance(i, SDFObject)
    assert "opI(" in i.to_glsl()

def test_difference(shapes):
    s, b = shapes
    d = s - b
    assert isinstance(d, SDFObject)
    assert "opS(" in d.to_glsl()

def test_xor(shapes):
    s, b = shapes
    x = s.xor(b)
    assert isinstance(x, SDFObject)
    assert "opX(" in x.to_glsl()

def test_smooth_union(shapes):
    s, b = shapes
    su = s.smooth_union(b, k=0.2)
    assert isinstance(su, SDFObject)
    assert "sUnion(" in su.to_glsl()

def test_smooth_intersection(shapes):
    s, b = shapes
    si = s.smooth_intersection(b, k=0.2)
    assert isinstance(si, SDFObject)
    assert "sIntersect(" in si.to_glsl()

def test_smooth_difference(shapes):
    s, b = shapes
    sd = s.smooth_difference(b, k=0.2)
    assert isinstance(sd, SDFObject)
    assert "sDifference(" in sd.to_glsl()

def test_round(shapes):
    s, _ = shapes
    r = s.round(0.1)
    assert isinstance(r, SDFObject)
    assert "opRound(" in r.to_glsl()

def test_bevel(shapes):
    s, _ = shapes
    o = s.bevel(0.1)
    assert isinstance(o, SDFObject)
    assert "opBevel(" in o.to_glsl()

def test_displace(shapes):
    s, _ = shapes
    disp = "sin(p.x * 10.0) * 0.1"
    d = s.displace(disp)
    assert isinstance(d, SDFObject)
    assert f"opDisplace({s.to_glsl()}, {disp})" in d.to_glsl()

def test_extrude():
    circle_2d = Forge("return length(p.xy) - 1.0;")
    extruded = circle_2d.extrude(0.5)
    assert isinstance(extruded, SDFObject)
    assert "opExtrude(" in extruded.to_glsl()

def test_color(shapes):
    s, _ = shapes
    colored_sphere = s.color(1, 0, 0)
    assert isinstance(colored_sphere, SDFObject)
    assert colored_sphere.color == (1, 0, 0)

def test_union_callable(shapes):
    s, b = shapes
    u_callable = (s | b).to_callable()
    points = np.array([[0, 0, 0], [0.8, 0, 0], [1.2, 0, 0]])
    s_dist = s.to_callable()(points)
    b_dist = b.to_callable()(points)
    expected = np.minimum(s_dist, b_dist)
    assert np.allclose(u_callable(points), expected)

def test_intersection_callable(shapes):
    s, b = shapes
    i_callable = (s & b).to_callable()
    points = np.array([[0, 0, 0], [0.8, 0, 0], [1.2, 0, 0]])
    s_dist = s.to_callable()(points)
    b_dist = b.to_callable()(points)
    expected = np.maximum(s_dist, b_dist)
    assert np.allclose(i_callable(points), expected)

def test_difference_callable(shapes):
    s, b = shapes
    d_callable = (s - b).to_callable()
    points = np.array([[0, 0, 0], [0.8, 0, 0], [1.2, 0, 0]])
    s_dist = s.to_callable()(points)
    b_dist = b.to_callable()(points)
    expected = np.maximum(s_dist, -b_dist)
    assert np.allclose(d_callable(points), expected)

def test_xor_callable(shapes):
    s, b = shapes
    x_callable = s.xor(b).to_callable()
    points = np.array([[0, 0, 0], [0.8, 0, 0], [1.2, 0, 0]])
    s_dist = s.to_callable()(points)
    b_dist = b.to_callable()(points)
    expected = np.maximum(np.minimum(s_dist, b_dist), -np.maximum(s_dist, b_dist))
    assert np.allclose(x_callable(points), expected)

def test_round_callable(shapes):
    s = shapes[0]
    r_callable = s.round(0.2).to_callable()
    points = np.array([[0.5, 0, 0], [1.0, 0, 0]])
    expected = np.array([-0.7, -0.2])
    assert np.allclose(r_callable(points), expected)

def test_bevel_callable(shapes):
    s = shapes[0]
    o_callable = s.bevel(0.1).to_callable()
    points = np.array([[0.5, 0, 0], [1.0, 0, 0], [1.2, 0, 0]])
    s_dist = s.to_callable()(points)
    expected = np.abs(s_dist) - 0.1
    assert np.allclose(o_callable(points), expected)

def test_smooth_union_callable(shapes):
    s, b = shapes
    su_callable = s.smooth_union(b, k=0.5).to_callable()
    point = np.array([[0.875, 0, 0]])
    s_dist = s.to_callable()(point)
    b_dist = b.to_callable()(point)
    assert su_callable(point)[0] < -0.125

def test_smooth_intersection_callable(shapes):
    s, b = shapes
    k = 0.5
    si_callable = s.smooth_intersection(b, k=k).to_callable()
    points = np.array([[0.8, 0, 0]])
    d1 = s.to_callable()(points)
    d2 = b.to_callable()(points)
    h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
    expected = d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h)
    assert np.allclose(si_callable(points), expected)

def test_smooth_difference_callable(shapes):
    s, b = shapes
    k = 0.5
    sd_callable = s.smooth_difference(b, k=k).to_callable()
    points = np.array([[0.8, 0, 0]])
    d1 = s.to_callable()(points)
    d2 = -b.to_callable()(points)
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    expected = d1 * (1.0 - h) + d2 * h + k * h * (1.0 - h)
    assert np.allclose(sd_callable(points), expected)

def test_extrude_callable():
    circle_callable = lambda p: np.linalg.norm(p[:, [0, 1]], axis=-1) - 1.0
    class Circle2D(SDFObject):
        def to_callable(self):
            return circle_callable
        def to_glsl(self):
            return ""

    h = 0.5
    extruded_callable = Circle2D().extrude(h).to_callable()
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0.5],
        [1, 0, 0.5],
        [0, 0, 1.0]
    ])
    d = circle_callable(points)
    w = np.stack([d, np.abs(points[:, 2]) - h], axis=-1)
    expected = np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=-1)
    assert np.allclose(extruded_callable(points), expected)