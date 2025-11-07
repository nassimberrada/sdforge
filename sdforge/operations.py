import numpy as np
from functools import reduce
from .core import SDFObject, _get_glsl_content, _glsl_format

# --- Standard Operations ---

class Union(SDFObject):
    """Represents the union of multiple SDF objects."""
    def __init__(self, *children, k=0.0):
        super().__init__()
        self.children = children
        self.k = k
    def to_glsl(self) -> str:
        op = "sUnion" if self.k and self.k > 0 else "opU"
        params = f", {_glsl_format(self.k)}" if op == "sUnion" else ""
        return reduce(lambda a, b: f"{op}({a}, {b}{params})", [c.to_glsl() for c in self.children])
    def to_callable(self):
        callables = [c.to_callable() for c in self.children]
        if self.k == 0.0:
            return lambda p: reduce(np.minimum, [c(p) for c in callables])
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        k = float(self.k)
        def _callable(p):
            dists = [c(p) for c in callables]
            res = dists[0]
            for i in range(1, len(dists)):
                d1, d2 = res, dists[i]
                h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
                res = d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)
            return res
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + sum([c.get_glsl_definitions() for c in self.children], [])
    def _collect_materials(self, materials):
        for c in self.children: c._collect_materials(materials)

class Intersection(SDFObject):
    """Represents the intersection of multiple SDF objects."""
    def __init__(self, *children, k=0.0):
        super().__init__()
        self.children = children
        self.k = k
    def to_glsl(self) -> str:
        op = "sIntersect" if self.k and self.k > 0 else "opI"
        params = f", {_glsl_format(self.k)}" if op == "sIntersect" else ""
        return reduce(lambda a, b: f"{op}({a}, {b}{params})", [c.to_glsl() for c in self.children])
    def to_callable(self):
        callables = [c.to_callable() for c in self.children]
        if self.k == 0.0:
            return lambda p: reduce(np.maximum, [c(p) for c in callables])
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        k = float(self.k)
        def _callable(p):
            dists = [c(p) for c in callables]
            res = dists[0]
            for i in range(1, len(dists)):
                d1, d2 = res, dists[i]
                h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
                res = d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h)
            return res
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + sum([c.get_glsl_definitions() for c in self.children], [])
    def _collect_materials(self, materials):
        for c in self.children: c._collect_materials(materials)

class Difference(SDFObject):
    """Represents the subtraction of one SDF object from another."""
    def __init__(self, a, b, k=0.0):
        super().__init__()
        self.a, self.b, self.k = a, b, k
    def to_glsl(self) -> str:
        op = "sDifference" if self.k and self.k > 0 else "opS"
        params = f", {_glsl_format(self.k)}" if op == "sDifference" else ""
        return f"{op}({self.a.to_glsl()}, {self.b.to_glsl()}{params})"
    def to_callable(self):
        a_call, b_call = self.a.to_callable(), self.b.to_callable()
        if self.k == 0.0:
            return lambda p: np.maximum(a_call(p), -b_call(p))
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        k = float(self.k)
        def _callable(p):
            d1, d2 = a_call(p), -b_call(p)
            h = np.clip(0.5 - 0.5 * (d1 - d2) / k, 0.0, 1.0)
            return d1 * (1.0 - h) + d2 * h + k * h * (1.0 - h)
        return _callable
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