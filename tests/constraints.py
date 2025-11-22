import pytest
import numpy as np
from unittest.mock import patch
from sdforge import box, cylinder, sphere, X, Y, Z
from sdforge.api.primitives import Box
from sdforge.api.constraints import (
    coincident,
    tangent_offset,
    midpoint,
)
from sdforge.api.transforms import Translate, Rotate

def test_coincident():
    p1 = (1, 2, 3)
    p2 = (1.00000001, 2.0, 3.0)
    p3 = (1, 2, 4)
    assert coincident(p1, p2) is True
    assert coincident(p1, p3) is False

def test_midpoint():
    p1 = np.array([0, 0, 0])
    p2 = np.array([10, 20, -30])
    m = midpoint(p1, p2)
    assert np.allclose(m, [5, 10, -15])

def test_tangent_offset():
    # Line along X axis
    offset = tangent_offset(circle_radius=5.0, line_direction=X)
    assert np.allclose(offset, [0, 5, 0])

    # Line along Y axis
    offset = tangent_offset(circle_radius=3.0, line_direction=Y)
    assert np.allclose(offset, [-3, 0, 0])

    # Line at 45 degrees
    direction = np.array([1, 1, 0]) / np.sqrt(2)
    offset = tangent_offset(circle_radius=1.0, line_direction=direction)
    # Perpendicular vector is (-1, 1, 0) / sqrt(2)
    expected = np.array([-1, 1, 0]) / np.sqrt(2)
    assert np.allclose(offset, expected)

def test_align_to_fluent_cardinal_axes():
    """Tests that .align_to() produces the correct sequence of transforms."""
    c = cylinder(radius=0.1, height=1.0)
    origin_point = np.array([0,0,0])

    # Align to +X face using fluent method
    aligned_x = c.align_to(reference_point=origin_point, face_normal=X)
    assert isinstance(aligned_x, Translate)
    assert np.allclose(aligned_x.offset, [0.5, 0, 0])
    assert isinstance(aligned_x.child, Rotate)
    assert np.allclose(aligned_x.child.axis, Z)

    # Align to +Y face (should only translate)
    aligned_y = c.align_to(reference_point=origin_point, face_normal=Y)
    assert isinstance(aligned_y, Translate)
    assert np.allclose(aligned_y.offset, [0, 0.5, 0])
    assert not isinstance(aligned_y.child, Rotate) # No rotation needed

    # Align to -Z face
    aligned_neg_z = c.align_to(reference_point=origin_point, face_normal=-Z)
    assert isinstance(aligned_neg_z, Translate)
    assert np.allclose(aligned_neg_z.offset, [0, 0, -0.5])
    assert isinstance(aligned_neg_z.child, Rotate)
    assert np.allclose(aligned_neg_z.child.axis, X)
    assert np.isclose(aligned_neg_z.child.angle, -np.pi / 2)

def test_align_to_fluent_with_offset():
    c = cylinder(radius=0.1, height=1.0)
    origin_point = np.array([0,0,0])
    # Use fluent method with offset
    aligned_x = c.align_to(reference_point=origin_point, face_normal=X, offset=2.0)

    # Should be at half-height (0.5) + offset (2.0)
    assert np.allclose(aligned_x.offset, [2.5, 0, 0])

def test_align_to_fluent_non_cardinal_axis_fails():
    c = cylinder(radius=0.1, height=1.0)
    with pytest.raises(ValueError, match="only supports cardinal axis normals"):
        c.align_to(reference_point=np.array([0,0,0]), face_normal=np.array([1,1,0]))

def test_place_at_angle_fluent():
    """Tests the transform chain created by .place_at_angle()."""
    b = box(0.1)
    pivot = np.array([10, 20, 30])
    angle = np.pi / 4
    distance = 5.0

    placed_obj = b.place_at_angle(pivot, axis=Y, angle_rad=angle, distance=distance)

    # 1. Final node is translation to the pivot point
    assert isinstance(placed_obj, Translate)
    assert np.allclose(placed_obj.offset, pivot)

    # 2. Its child is the rotation
    rot_node = placed_obj.child
    assert isinstance(rot_node, Rotate)
    assert np.allclose(rot_node.axis, Y)
    assert np.isclose(rot_node.angle, angle)

    # 3. Its child is the initial distance translation
    dist_node = rot_node.child
    assert isinstance(dist_node, Translate)
    assert np.allclose(dist_node.offset, X * distance)

    # 4. Final child is the original box
    assert dist_node.child == b

def test_offset_along_fluent():
    """Tests the .offset_along() fluent helper."""
    b = box(0.1)
    start_point = np.array([1, 2, 3])
    direction = np.array([0, 1, 0]) # Y-axis
    distance = 10.0

    offset_obj = b.offset_along(start_point, direction, distance)

    assert isinstance(offset_obj, Translate)
    expected_pos = [1, 12, 3]
    assert np.allclose(offset_obj.offset, expected_pos)
    assert offset_obj.child == b

def test_bounding_box_fluent():
    """Tests the .bounding_box() fluent helper using a mock."""
    s = sphere(1.5)

    # Mock the estimate_bounds method to return a predictable value
    mock_bounds = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
    with patch.object(s, 'estimate_bounds', return_value=mock_bounds) as mock_method:
        bbox = s.bounding_box(padding=0.1)

        mock_method.assert_called_once()

        # The resulting box should be centered at the origin
        assert isinstance(bbox, Translate)
        assert np.allclose(bbox.offset, [0,0,0])

        # Its size should be the diameter (3.0) + padding on both sides (0.2)
        assert isinstance(bbox.child, Box)
        assert np.allclose(bbox.child.size, (3.2, 3.2, 3.2))