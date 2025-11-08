import pytest
from sdforge.api.camera import Camera

def test_camera_instantiation_defaults():
    """Tests that the Camera initializes with correct default values."""
    cam = Camera()
    assert cam.position == (5, 4, 5)
    assert cam.target == (0, 0, 0)
    assert cam.zoom == 1.0

def test_camera_instantiation_custom_values():
    """Tests that the Camera correctly stores custom initialization values."""
    pos = (10, 20, 30)
    tgt = (1, 2, 3)
    zoom = 2.5
    cam = Camera(position=pos, target=tgt, zoom=zoom)
    assert cam.position == pos
    assert cam.target == tgt
    assert cam.zoom == zoom

def test_camera_attributes_are_mutable():
    """Ensures camera attributes can be changed after instantiation."""
    cam = Camera()
    cam.position = (1, 1, 1)
    cam.zoom = 5.0
    assert cam.position == (1, 1, 1)
    assert cam.zoom == 5.0