import numpy as np
from .primitives import line, curve
from .operations import Union
from ..core import SDFNode

class Sketch:
    """
    A builder class for creating 2D profiles using a path-based interface.

    The Sketch API allows you to define shapes by moving a 'pen' sequentially
    (move_to, line_to, curve_to). The result is a collection of connected
    segments (capsules and bezier tubes) that can be extruded or revolved.

    Note: The resulting SDF represents the *outline* (stroke) of the shape,
    defined by the `stroke_radius`. It does not currently create a filled polygon.
    """

    def __init__(self, start=(0.0, 0.0)):
        """
        Starts a new sketch.

        Args:
            start (tuple): The starting (x, y) coordinates of the pen.
        """
        self._segments = []
        # Ensure we work with float arrays
        self._current_pos = np.array(start, dtype=float)
        self._start_pos = self._current_pos.copy()

    def _to_vec3(self, p2d):
        """Helper to convert 2D (x,y) to 3D (x,y,0) for internal primitives."""
        if len(p2d) == 2:
            return np.array([p2d[0], p2d[1], 0.0])
        return np.array(p2d)

    def move_to(self, x, y):
        """
        Moves the pen to a new position without drawing a line.
        This effectively starts a new sub-path.

        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
        """
        self._current_pos = np.array([x, y], dtype=float)
        # If this is the very first operation, update start_pos too
        if not self._segments:
            self._start_pos = self._current_pos
        return self

    def line_to(self, x, y):
        """
        Draws a straight line from the current position to (x, y).

        Args:
            x (float): Target X coordinate.
            y (float): Target Y coordinate.
        """
        end_pos = np.array([x, y], dtype=float)

        # Create a closure/data struct to defer SDF creation until to_sdf()
        # We store 3D coordinates (z=0) because base primitives are 3D
        seg_data = {
            'type': 'line',
            'start': self._to_vec3(self._current_pos),
            'end': self._to_vec3(end_pos)
        }
        self._segments.append(seg_data)
        self._current_pos = end_pos
        return self

    def curve_to(self, x, y, control):
        """
        Draws a Quadratic Bezier curve from current position to (x, y).

        Args:
            x (float): Target X coordinate (end point).
            y (float): Target Y coordinate (end point).
            control (tuple): The (cx, cy) control point influencing the curve.
        """
        end_pos = np.array([x, y], dtype=float)
        ctrl_pos = np.array(control, dtype=float)

        seg_data = {
            'type': 'bezier',
            'start': self._to_vec3(self._current_pos),
            'control': self._to_vec3(ctrl_pos),
            'end': self._to_vec3(end_pos)
        }
        self._segments.append(seg_data)
        self._current_pos = end_pos
        return self

    def close(self):
        """
        Draws a line from the current position back to the starting position
        of the current sub-path.
        """
        # Avoid creating a zero-length segment
        if not np.allclose(self._current_pos, self._start_pos):
            self.line_to(self._start_pos[0], self._start_pos[1])
        return self

    def to_sdf(self, stroke_radius=0.05) -> SDFNode:
        """
        Converts the sketch into an SDF object.

        Args:
            stroke_radius (float): The thickness radius of the lines/curves.

        Returns:
            SDFNode: A union of all segments in the sketch.
        """
        nodes = []
        for seg in self._segments:
            if seg['type'] == 'line':
                # Use the primitive line() which handles to_profile_glsl correctly
                nodes.append(line(seg['start'], seg['end'], radius=stroke_radius))
            elif seg['type'] == 'bezier':
                nodes.append(curve(seg['start'], seg['control'], seg['end'], radius=stroke_radius))

        if not nodes:
            # Return an empty/invisible object if no segments
            from ..api.primitives import sphere
            return sphere(0).translate((1e5, 0, 0))

        # Return a Union of all segments
        return Union(nodes)