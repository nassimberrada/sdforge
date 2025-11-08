import pytest
from sdforge.api.light import Light

def test_light_instantiation():
    """Tests static instantiation of the Light class."""
    light_static = Light(position=(5, 5, 5), ambient_strength=0.2, shadow_softness=16.0, ao_strength=5.0)
    assert light_static.position == (5, 5, 5)
    assert light_static.ambient_strength == 0.2
    assert light_static.shadow_softness == 16.0
    assert light_static.ao_strength == 5.0