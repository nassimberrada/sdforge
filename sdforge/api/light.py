class Light:
    """Represents light and shadow properties for the scene."""
    def __init__(self, position=None, ambient_strength=0.1, shadow_softness=8.0, ao_strength=3.0):
        """
        Initializes the scene light.

        Args:
            position (tuple, optional): The position of the light source. If None,
                                        the light is positioned at the camera. Defaults to None.
            ambient_strength (float, optional): The minimum brightness for surfaces. Defaults to 0.1.
            shadow_softness (float, optional): How soft shadows are. Higher is softer. Defaults to 8.0.
            ao_strength (float, optional): Strength of ambient occlusion. Defaults to 3.0.
        """
        self.position = position
        self.ambient_strength = ambient_strength
        self.shadow_softness = shadow_softness
        self.ao_strength = ao_strength