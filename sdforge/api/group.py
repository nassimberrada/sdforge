import numpy as np
from functools import reduce
from ..core import SDFNode, GLSLContext
from ..utils import _glsl_format
from .params import Param
from .operations import Union

class Group(SDFNode):
    """
    A collection of SDF objects that can be transformed as a single unit.

    A Group acts like a container. Any transformation (like `.translate()` or
    `.rotate()`) applied to the group is propagated to all of its children.
    When rendered or meshed, the group evaluates as the union of all its
    child objects.
    """
    def __init__(self, *children):
        """
        Initializes a Group with one or more child objects.

        Args:
            *children (SDFNode): A variable number of SDFNode objects to include
                                 in the group.
        
        Example:
            >>> s = sphere(0.5).translate((-1, 0, 0))
            >>> b = box(0.5).translate((1, 0, 0))
            >>> g = Group(s, b)
            >>> scene = g.rotate(Y, np.pi / 4)
        """
        super().__init__()
        self.children = children

    def to_glsl(self, ctx: GLSLContext) -> str:
        """Returns the GLSL representation of the union of all children."""
        if not self.children:
            return "vec4(1e9, -1.0, 0.0, 0.0)"
        return Union(children=list(self.children)).to_glsl(ctx)

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        """Returns the GLSL representation of the union of all children's profiles."""
        if not self.children:
            return "vec4(1e9, -1.0, 0.0, 0.0)"
        return Union(children=list(self.children)).to_profile_glsl(ctx)

    def to_callable(self):
        """Returns a callable for the union of all children."""
        if not self.children:
            return lambda p: np.full(p.shape[0] if p.ndim > 1 else 1, 1e9)
        return Union(children=list(self.children)).to_callable()

    def to_profile_callable(self):
        """Returns a callable for the union of all children's profiles."""
        if not self.children:
            return lambda p: np.full(p.shape[0] if p.ndim > 1 else 1, 1e9)
        return Union(children=list(self.children)).to_profile_callable()

    def _apply_to_children(self, method_name, *args, **kwargs):
        """
        Applies a method to each child and returns a new Group with the results.
        """
        new_children = [getattr(child, method_name)(*args, **kwargs) for child in self.children]
        return Group(*new_children)

# List of all methods on SDFNode that return a new transformed/shaped SDFNode.
# These will be dynamically added to the Group class to propagate the operation.
_PROPAGATED_METHODS = [
    'translate', 'scale', 'rotate', 'orient', 'twist', 'bend',
    'repeat', 'limited_repeat', 'polar_repeat', 'mirror',
    'round', 'shell', 'bevel', 'extrude', 'revolve',
    'displace', 'displace_by_noise',
]

def _make_propagated_method(name):
    def method(self, *args, **kwargs):
        return self._apply_to_children(name, *args, **kwargs)
    try:
        method.__doc__ = getattr(SDFNode, name).__doc__
    except AttributeError:
        pass # Method might not have a docstring
    return method

for method_name in _PROPAGATED_METHODS:
    setattr(Group, method_name, _make_propagated_method(method_name))