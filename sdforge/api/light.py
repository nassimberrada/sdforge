class Light:
    """Represents light and shadow properties for the scene."""
    def __init__(self, position=None, ambient_strength=0.1, shadow_softness=8.0, ao_strength=3.0):
        """
        Initializes the scene light.

        Args:
            position (tuple, optional): The position of the light source. If None,
                                        the light is treated as a "headlight" attached
                                        to the camera. Defaults to None.
            ambient_strength (float, optional): The minimum brightness for surfaces,
                                                preventing shadows from being pure black.
                                                Defaults to 0.1.
            shadow_softness (float, optional): Controls how soft shadows are. Higher
                                               values produce softer shadows but can be
                                               more computationally expensive.
                                               Defaults to 8.0.
            ao_strength (float, optional): Strength of the ambient occlusion effect, which
                                           darkens crevices and contact points.
                                           Defaults to 3.0.
        
        Example:
            >>> from sdforge import sphere, Light
            >>> scene = sphere(1.0)
            >>> # Create a bright, hard-edged light from a fixed position
            >>> light = Light(position=(5, 5, 3), shadow_softness=1.0)
            >>> scene.render(light=light)
        """
        self.position = position
        self.ambient_strength = ambient_strength
        self.shadow_softness = shadow_softness
        self.ao_strength = ao_strength