class Debug:
    """
    Represents a debug visualization mode for the renderer.
    """
    def __init__(self, mode: str, plane: str = 'xy', slice_height: float = 0.0, view_scale: float = 4.0):
        """
        Initializes the debug mode object.

        Args:
            mode (str): The debug visualization to enable.
                        Supported options: 
                        - 'normals': Visualizes surface normals.
                        - 'steps': Visualizes raymarching cost (heatmap).
                        - 'slice': Visualizes a 2D cross-section of the field.

            plane (str, optional): The plane to slice through. Used only if mode='slice'.
                                   Options: 'xy', 'xz', 'yz'. Defaults to 'xy'.

            slice_height (float, optional): The offset of the slice plane from the origin.
                                            Used only if mode='slice'. Defaults to 0.0.

            view_scale (float, optional): The vertical size of the view in world units.
                                          Used only if mode='slice'. Defaults to 4.0.
        """
        self.mode = mode
        self.plane = plane.lower()
        self.slice_height = slice_height
        self.view_scale = view_scale