import uuid

class Param:
    """
    An interactive, real-time parameter for an SDF model.

    When a `Param` object is used as a value for an SDF primitive or
    operation, it will automatically generate a UI slider in the render
    window (if a UI backend like Dear ImGui is enabled). Dragging the
    slider will update the 3D model in real-time.
    """
    def __init__(self, name: str, default: float, min_val: float, max_val: float):
        """
        Initializes an interactive parameter.

        Args:
            name (str): The display name for the slider in the UI.
            default (float): The initial value of the parameter.
            min_val (float): The minimum value of the slider range.
            max_val (float): The maximum value of the slider range.
        """
        self.name = name
        self.value = default
        self.min_val = min_val
        self.max_val = max_val
        # Sanitize name for use as a GLSL variable
        sanitized_name = ''.join(c if c.isalnum() else '_' for c in name)
        self.uniform_name = f"u_param_{sanitized_name}_{uuid.uuid4().hex[:6]}"

    def __str__(self):
        """Returns the GLSL uniform name for use in shader code."""
        return self.uniform_name

    def to_glsl(self):
        """Returns the GLSL uniform name."""
        return self.uniform_name