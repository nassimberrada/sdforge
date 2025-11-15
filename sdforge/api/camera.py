class Camera:
    """Represents a camera in the scene for static positioning."""

    def __init__(self, position=(5, 4, 5), target=(0, 0, 0), zoom=1.0):
        """
        Initializes the camera.

        When a Camera object is passed to the `.render()` method, the view
        is fixed to these settings. If no camera is provided, an interactive
        orbit camera is used instead.

        Args:
            position (tuple, optional): The position of the camera in 3D space.
                                        Defaults to (5, 4, 5).
            target (tuple, optional): The point the camera is looking at.
                                      Defaults to (0, 0, 0).
            zoom (float, optional): The zoom level. Higher is more zoomed in.
                                    Defaults to 1.0.
        
        Example:
            >>> from sdforge import sphere, Camera
            >>> scene = sphere(1.0)
            >>> # Create a camera positioned far away, looking at the origin
            >>> cam = Camera(position=(10, 8, 10), target=(0, 0, 0))
            >>> scene.render(camera=cam)
        """
        self.position = position
        self.target = target
        self.zoom = zoom