# camera.py
class Camera:
    """Represents a camera in the scene for static or interactive positioning."""

    def __init__(self, position=(5, 4, 5), target=(0, 0, 0), zoom=1.0, type='perspective'):
        """
        Initializes the camera.

        When a Camera object is passed to the `.render()` method, the view
        is fixed to these settings unless `interactive=True` is provided. 
        If no camera is provided, an interactive perspective orbit camera is used.

        Args:
            position (tuple, optional): The position of the camera in 3D space. Defaults to (5, 4, 5).
                                        Note: (5, 4, 5) naturally creates an isometric angle.
            target (tuple, optional): The point the camera is looking at. Defaults to (0, 0, 0).
            zoom (float, optional): The zoom level. Defaults to 1.0.
            type (str, optional): Projection type, 'perspective' or 'orthographic'. 
                                  Defaults to 'perspective'. Use 'orthographic' for isometric views.
        
        Example:
            >>> from sdforge import sphere, Camera
            >>> scene = sphere(1.0)
            >>> iso_cam = Camera(type='orthographic')
            >>> scene.render(camera=iso_cam)
        """
        self.position = position
        self.target = target
        self.zoom = zoom
        self.type = type