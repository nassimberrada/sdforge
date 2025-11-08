class Camera:
    """Represents a camera in the scene for static positioning."""

    def __init__(self, position=(5, 4, 5), target=(0, 0, 0), zoom=1.0):
        """
        Initializes the camera.

        Args:
            position (tuple, optional): The position of the camera in 3D space.
                                        Defaults to (5, 4, 5).
            target (tuple, optional): The point the camera is looking at.
                                      Defaults to (0, 0, 0).
            zoom (float, optional): The zoom level. Higher is more zoomed in.
                                    Defaults to 1.0.
        """
        self.position = position
        self.target = target
        self.zoom = zoom