import numpy as np
from functools import reduce
from .core import SDFObject, _get_glsl_content, _glsl_format

# --- Standard Operations ---

class Union(SDFObject):
    """Represents the union of multiple SDF objects."""
    def __init__(self, *children):
        super().__init__()
        self.children = children
    def to_glsl(self) -> str: return reduce(lambda a, b: f"opU({a}, {b})", [c.to_glsl() for c in self.children])
    def to_callable(self):
        callables = [c.to_callable() for c in self.children]; return lambda p: reduce(np.minimum, [c(p) for c in callables])
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + sum([c.get_glsl_definitions() for c in self.children], [])
    def _collect_materials(self, materials):
        for c in self.children: c._collect_materials(materials)

class Intersection(SDFObject):
    """Represents the intersection of multiple SDF objects."""
    def __init__(self, *children):
        super().__init__()
        self.children = children
    def to_glsl(self) -> str: return reduce(lambda a, b: f"opI({a}, {b})", [c.to_glsl() for c in self.children])
    def to_callable(self):
        callables = [c.to_callable() for c in self.children]; return lambda p: reduce(np.maximum, [c(p) for c in callables])
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + sum([c.get_glsl_definitions() for c in self.children], [])
    def _collect_materials(self, materials):
        for c in self.children: c._collect_materials(materials)

class Difference(SDFObject):
    """Represents the subtraction of one SDF object from another."""
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b
    def to_glsl(self) -> str: return f"opS({self.a.to_glsl()}, {self.b.to_glsl()})"
    def to_callable(self):
        a_call, b_call = self.a.to_callable(), self.b.to_callable(); return lambda p: np.maximum(a_call(p), -b_call(p))
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)

class Xor(SDFObject):
    """Represents the exclusive-or (XOR) of two SDF objects."""
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b
    def to_glsl(self) -> str: return f"opX({self.a.to_glsl()}, {self.b.to_glsl()})"
    def to_callable(self):
        a_call, b_call = self.a.to_callable(), self.b.to_callable(); return lambda p: np.maximum(np.minimum(a_call(p), b_call(p)), -np.maximum(a_call(p), b_call(p)))
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)


# --- Smooth Operations ---

class SmoothUnion(SDFObject):
    """Represents the smooth union of two SDF objects."""
    def __init__(self, a, b, k):
        super().__init__()
        self.a, self.b, self.k = a, b, k
    def to_glsl(self) -> str: return f"sUnion({self.a.to_glsl()}, {self.b.to_glsl()}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        a_call, b_call, k = self.a.to_callable(), self.b.to_callable(), float(self.k)
        def _callable(p):
            d1, d2 = a_call(p), b_call(p); h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
            return d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)

class SmoothIntersection(SDFObject):
    """Represents the smooth intersection of two SDF objects."""
    def __init__(self, a, b, k):
        super().__init__()
        self.a, self.b, self.k = a, b, k
    def to_glsl(self) -> str: return f"sIntersect({self.a.to_glsl()}, {self.b.to_glsl()}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        a_call, b_call, k = self.a.to_callable(), self.b.to_callable(), float(self.k)
        def _callable(p):
            d1, d2 = a_call(p), b_call(p)
            h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
            return d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)

class SmoothDifference(SDFObject):
    """Represents the smooth difference of two SDF objects."""
    def __init__(self, a, b, k):
        super().__init__()
        self.a, self.b, self.k = a, b, k
    def to_glsl(self) -> str: return f"sDifference({self.a.to_glsl()}, {self.b.to_glsl()}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        a_call, b_call, k = self.a.to_callable(), self.b.to_callable(), float(self.k)
        def _callable(p):
            d1, d2 = a_call(p), -b_call(p)
            h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
            return d1 * (1.0 - h) + d2 * h + k * h * (1.0 - h)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)
