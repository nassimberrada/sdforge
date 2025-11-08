class Debug:
    """
    Represents a debug visualization mode for the renderer.
    """
    def __init__(self, mode: str):
        """
        Initializes the debug mode object.

        Args:
            mode (str): The debug visualization to enable.
                        Supported options: 'normals', 'steps'.
        """
        self.mode = mode