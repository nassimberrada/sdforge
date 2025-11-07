import pytest
import numpy as np
from sdforge import SDFObject, sphere, box, Forge, Group, Material

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

def test_smooth_union_method(shapes):
    s, b = shapes
    su = s.union(b, k=0.2)
    assert isinstance(su, SDFObject)
    assert "sUnion(" in su.to_glsl()

def test_smooth_intersection_method(shapes):
    s, b = shapes
    si = s.intersection(b, k=0.2)
    assert isinstance(si, SDFObject)
    assert "sIntersect(" in si.to_glsl()

def test_smooth_difference_method(shapes):
    s, b = shapes
    sd = s.difference(b, k=0.2)
    assert isinstance(sd, SDFObject)
    assert "sDifference(" in sd.to_glsl()

def test_round(shapes):
    s, _ = shapes
    r = s.round(0.1)
    assert isinstance(r, SDFObject)
    assert "opRound(" in r.to_glsl()

def test_shell_and_bevel(shapes):
    s, _ = shapes
    o = s.shell(0.1)
    assert isinstance(o, SDFObject)
    assert "opBevel(" in o.to_glsl()
    # Test alias
    o2 = s.bevel(0.1)
    assert o.to_glsl() == o2.to_glsl()

def test_displace(shapes):
    s, _ = shapes
    disp = "sin(p.x * 10.0) * 0.1"
    d = s.displace(disp)
    assert isinstance(d, SDFObject)
    assert f"opDisplace({s.to_glsl()}, {disp})" in d.to_glsl()

def test_displace_by_noise(shapes):
    s, _ = shapes
    d = s.displace_by_noise(scale=5.0, strength=0.2)
    assert isinstance(d, SDFObject)
    assert "snoise(p * 5.0) * 0.2" in d.to_glsl()
    assert any("snoise" in definition for definition in d.get_glsl_definitions())

def test_extrude():
    circle_2d = Forge("return length(p.xy) - 1.0;")
    extruded = circle_2d.extrude(0.5)
    assert isinstance(extruded, SDFObject)
    assert "opExtrude(" in extruded.to_glsl()

def test_revolve():
    profile_2d = Forge("return length(p.xy - vec2(1.0, 0.0)) - 0.2;")
    revolved = profile_2d.revolve()
    assert isinstance(revolved, SDFObject)
    assert "vec2(length(p.xz), p.y)" in revolved.to_glsl()


def test_color(shapes):
    s, _ = shapes
    colored_sphere = s.color(1, 0, 0)
    assert isinstance(colored_sphere, SDFObject)
    # The color property is on the Material object, which wraps the sphere
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

def test_shell_callable(shapes):
    s = shapes[0]
    o_callable = s.shell(0.1).to_callable()
    points = np.array([[0.5, 0, 0], [1.0, 0, 0], [1.2, 0, 0]])
    s_dist = s.to_callable()(points)
    expected = np.abs(s_dist) - 0.1
    assert np.allclose(o_callable(points), expected)

def test_smooth_union_callable(shapes):
    s, b = shapes
    su_callable = s.union(b, k=0.5).to_callable()
    point = np.array([[0.875, 0, 0]])
    s_dist = s.to_callable()(point)
    b_dist = b.to_callable()(point)
    assert su_callable(point)[0] < -0.125

def test_smooth_intersection_callable(shapes):
    s, b = shapes
    k = 0.5
    si_callable = s.intersection(b, k=k).to_callable()
    points = np.array([[0.8, 0, 0]])
    d1 = s.to_callable()(points)
    d2 = b.to_callable()(points)
    h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
    expected = d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h)
    assert np.allclose(si_callable(points), expected)

def test_smooth_difference_callable(shapes):
    s, b = shapes
    k = 0.5
    sd_callable = s.difference(b, k=k).to_callable()
    points = np.array([[0.8, 0, 0]])
    d1 = s.to_callable()(points)
    d2 = -b.to_callable()(points)
    h = np.clip(0.5 - 0.5 * (d1 - d2) / k, 0.0, 1.0)
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

def test_revolve_callable():
    profile_callable = lambda p: np.linalg.norm(p[:, [0, 1]] - np.array([1.0, 0.0]), axis=-1) - 0.2
    class Profile2D(SDFObject):
        def to_callable(self): return profile_callable
        def to_glsl(self): return ""
    
    revolved_callable = Profile2D().revolve().to_callable()
    
    # Point on the surface of the revolved torus
    point_3d = np.array([[1.2, 0.0, 0.0]])
    # Corresponding 2D point for the profile
    point_2d = np.array([[1.2, 0.0, 0.0]])

    expected = profile_callable(point_2d) # Should be 0.0
    assert np.allclose(revolved_callable(point_3d), expected)

def test_group_creation(shapes):
    s, b = shapes
    g = Group(s, b)
    assert isinstance(g, SDFObject)
    assert len(g.children) == 2

def test_group_to_glsl_is_union(shapes):
    s, b = shapes
    g = Group(s, b)
    u = s | b
    # Group GLSL should be equivalent to a Union
    assert g.to_glsl() == u.to_glsl()

def test_group_transform_propagation(shapes):
    s, b = shapes
    g = Group(s, b)
    offset = (1, 2, 3)
    
    # Transform the group
    g_translated = g.translate(offset)
    
    # Manually transform children and create a new group
    s_translated = s.translate(offset)
    b_translated = b.translate(offset)
    expected_group = Group(s_translated, b_translated)
    
    assert isinstance(g_translated, Group)
    assert g_translated.to_glsl() == expected_group.to_glsl()

def test_group_chaining_transforms(shapes):
    s, b = shapes
    g = Group(s, b)
    
    g_transformed = g.translate((1,0,0)).scale(2.0)
    
    s_transformed = s.translate((1,0,0)).scale(2.0)
    b_transformed = b.translate((1,0,0)).scale(2.0)
    expected_group = Group(s_transformed, b_transformed)
    
    assert g_transformed.to_glsl() == expected_group.to_glsl()

def test_group_shaping_propagation(shapes):
    s, b = shapes
    g = Group(s, b)
    
    g_rounded = g.round(0.1)
    
    s_rounded = s.round(0.1)
    b_rounded = b.round(0.1)
    expected_group = Group(s_rounded, b_rounded)
    
    assert g_rounded.to_glsl() == expected_group.to_glsl()

def test_empty_group():
    g = Group()
    assert "1e9" in g.to_glsl()
    assert g.to_callable()(np.array([[0,0,0]]))[0] > 1e8

def test_bounded_by_is_intersection(shapes):
    s, b = shapes
    
    bounded_shape = s.bounded_by(b)
    intersect_shape = s & b

    assert bounded_shape.to_glsl() == intersect_shape.to_glsl()
    
    points = np.array([[0.8, 0, 0], [1.2, 0, 0]])
    assert np.allclose(bounded_shape.to_callable()(points), intersect_shape.to_callable()(points))