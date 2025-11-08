import numpy as np
from abc import ABC, abstractmethod

class GLSLContext:
    """Manages the state of the GLSL compilation process for a scene."""
    def __init__(self, compiler):
        self.compiler = compiler
        self.p = "p"  # The name of the current point variable being evaluated
        self.statements = []
        self.dependencies = set()
        self._var_counter = 0

    def add_statement(self, line: str):
        """Adds a line of code to the current function body."""
        self.statements.append(line)

    def new_variable(self, type: str, expression: str) -> str:
        """Declares a new GLSL variable and returns its name."""
        name = f"var_{self._var_counter}"
        self._var_counter += 1
        self.add_statement(f"{type} {name} = {expression};")
        return name

    def with_p(self, new_p_name: str) -> 'GLSLContext':
        """Creates a sub-context for a child node with a transformed point."""
        new_ctx = GLSLContext(self.compiler)
        new_ctx.p = new_p_name
        new_ctx.dependencies = self.dependencies
        new_ctx._var_counter = self._var_counter
        return new_ctx

class SDFNode(ABC):
    """Abstract base class for all SDF objects in the scene graph."""

    @abstractmethod
    def to_glsl(self, ctx: GLSLContext) -> str:
        """
        Contributes to the GLSL compilation and returns the name of the
        GLSL variable holding the vec4 result (dist, mat_id, 0, 0).
        """
        raise NotImplementedError

    @abstractmethod
    def to_callable(self):
        """
        Returns a Python function that takes a NumPy array of points (N, 3)
        and returns an array of distances (N,).
        """
        raise NotImplementedError

    def render(self, **kwargs):
        """Renders the SDF object in a live-updating viewer."""
        from .engine import render as render_func
        render_func(self, **kwargs)

    # --- Stubs for future functionality ---
    def translate(self, offset): raise NotImplementedError("Transforms not implemented yet.")
    def union(self, other): raise NotImplementedError("Operations not implemented yet.")
    def color(self, r, g, b): raise NotImplementedError("Materials not implemented yet.")