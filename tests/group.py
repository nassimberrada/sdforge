import pytest
import numpy as np
from sdforge import Group, sphere, box, X, Y
from sdforge.api.operations import Union
from sdforge.api.transforms import Translate

@pytest.fixture
def shapes():
    """Provides two basic shapes for grouping tests."""
    return sphere(r=1.0), box(size=1.5)

def test_group_acts_as_union(shapes):
    """Tests that a Group's GLSL and callable are equivalent to a Union."""
    s, b = shapes
    g = Group(s, b)
    u = Union([s, b])

    # This is a bit of an implementation detail test, but it's a good sanity check.
    # We can't compare GLSL strings directly due to variable names, but we can
    # test the callable, which is a pure numerical function.
    points = np.random.rand(100, 3)
    assert np.allclose(g.to_callable()(points), u.to_callable()(points))

def test_empty_group():
    """Tests that an empty group is valid and returns infinite distance."""
    g = Group()
    
    points = np.array([[0., 0., 0.], [10., 20., 30.]])
    distances = g.to_callable()(points)
    assert np.all(distances > 1e8)

def test_transform_propagation(shapes):
    """
    Tests that calling a transform method on a Group returns a new Group
    where each child has been transformed.
    """
    s, b = shapes
    g = Group(s, b)
    
    offset = (1, 2, 3)
    translated_group = g.translate(offset)
    
    # Check that the new object is still a Group
    assert isinstance(translated_group, Group)
    
    # Check that the children of the new group are now Translate nodes
    assert len(translated_group.children) == 2
    assert isinstance(translated_group.children[0], Translate)
    assert isinstance(translated_group.children[1], Translate)
    
    # Check that the original shapes were wrapped correctly
    assert translated_group.children[0].child == s
    assert translated_group.children[1].child == b

@pytest.mark.usefixtures("assert_equivalence")
def test_group_equivalence(assert_equivalence):
    """
    Tests numeric equivalence between Python and GLSL for a transformed group.
    """
    s = sphere(0.5).translate(-X)
    b = box(0.5).translate(X)
    g = Group(s, b)
    
    # A complex scene involving a transformed group
    scene = g.rotate(Y, 0.785).scale(1.5)
    
    assert_equivalence(scene)