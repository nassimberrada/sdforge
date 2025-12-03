import numpy as np
from .core import SDFNode, GLSLContext
from .operations import Union

class Group(SDFNode):
    """
    A collection of SDF objects that can be transformed as a single unit.
    """
    def __init__(self, *children):
        """
        Initializes a Group with one or more child objects.

        Args:
            *children (SDFNode): A variable number of SDFNode objects to include
                                 in the group.        
        """
        super().__init__()
        self.children = children

    def to_glsl(self, ctx: GLSLContext) -> str:
        if not self.children:
            return "vec4(1e9, -1.0, 0.0, 0.0)"
        return Union(children=list(self.children)).to_glsl(ctx)

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        if not self.children:
            return "vec4(1e9, -1.0, 0.0, 0.0)"
        return Union(children=list(self.children)).to_profile_glsl(ctx)

    def _apply_to_children(self, method_name, *args, **kwargs):
        new_children = [getattr(child, method_name)(*args, **kwargs) for child in self.children]
        return Group(*new_children)

_PROPAGATED_METHODS = [
    'translate', 'scale', 'rotate', 'orient', 'twist', 'bend',
    'repeat', 'mirror',
    'round', 'shell', 'extrude', 'revolve',
    'displace', 'displace_by_noise',
]

def _make_propagated_method(name):
    def method(self, *args, **kwargs):
        return self._apply_to_children(name, *args, **kwargs)
    try:
        method.__doc__ = getattr(SDFNode, name).__doc__
    except AttributeError:
        pass
    return method

for method_name in _PROPAGATED_METHODS:
    setattr(Group, method_name, _make_propagated_method(method_name))