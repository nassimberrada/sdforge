import numpy as np
from functools import reduce
from .core import SDFNode, GLSLContext
from .utils import _glsl_format, _smoothstep, Param

class Compositor(SDFNode):
    """
    A generic node for combining multiple SDFs.
    Unifies Boolean operations (Union, Intersection, Difference), Morphing, and Groups.
    """
    glsl_dependencies = {"compositors"}

    _GLSL_OPS = {
        'union': {'hard': 'opU', 'smooth': 'sUnion', 'linear': 'cUnion'},
        'intersection': {'hard': 'opI', 'smooth': 'sIntersect', 'linear': 'cIntersect'},
        'difference': {'hard': 'opS', 'smooth': 'sDifference', 'linear': 'cDifference'},
        'morph': {'default': 'opMorph'}
    }

    def __init__(self, children: list, op_type: str = 'union', blend: float = 0.0, blend_type: str = 'smooth', mask: SDFNode = None, mask_falloff: float = 0.0):
        super().__init__()
        self.children = children
        self.op_type = op_type.lower()
        self.blend = blend
        self.blend_type = blend_type
        self.mask = mask
        self.mask_falloff = mask_falloff

        if self.op_type not in self._GLSL_OPS:
            raise ValueError(f"Unknown operation type: {self.op_type}")

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)

        if profile_mode:
            child_vars = [c.to_profile_glsl(ctx) for c in self.children]
        else:
            child_vars = [c.to_glsl(ctx) for c in self.children]

        if not child_vars:
            return "vec4(1e9, -1.0, 0.0, 0.0)"
        
        if len(child_vars) == 1:
            return child_vars[0]

        if self.op_type == 'morph':

            blend_expr = _glsl_format(self.blend)
            if self.mask:
                mask_var = self.mask.to_glsl(ctx)
                falloff_str = _glsl_format(self.mask_falloff)
                factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
                blend_expr = f"({blend_expr} * {factor_expr})"
            
            func_name = self._GLSL_OPS['morph']['default']
            op_func = lambda a, b: f"{func_name}({a}, {b}, {blend_expr})"
            result_expr = reduce(op_func, child_vars)
            return ctx.new_variable('vec4', result_expr)

        is_blending = (isinstance(self.blend, (int, float)) and self.blend > 1e-6) or isinstance(self.blend, (str, Param))

        if is_blending:
            func_name = self._GLSL_OPS[self.op_type].get(self.blend_type, 'sUnion')
            blend_expr = _glsl_format(self.blend)
            if self.mask:
                mask_var = self.mask.to_glsl(ctx)
                falloff_str = _glsl_format(self.mask_falloff)
                factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
                blend_expr = f"({blend_expr} * {factor_expr})"

            op_func = lambda a, b: f"{func_name}({a}, {b}, {blend_expr})"
        else:
            func_name = self._GLSL_OPS[self.op_type]['hard']
            op_func = lambda a, b: f"{func_name}({a}, {b})"

        result_expr = reduce(op_func, child_vars)
        return ctx.new_variable('vec4', result_expr)

    def to_glsl(self, ctx: GLSLContext) -> str:
        return self._base_to_glsl(ctx, profile_mode=False)

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        return self._base_to_glsl(ctx, profile_mode=True)

    def _make_callable(self, child_callables):
        if not child_callables:
            return lambda p: np.full(len(p), 1e9)

        def check_dynamic(val):
            if isinstance(val, (str, Param)): return True
            if isinstance(val, (list, tuple, np.ndarray)):
                return any(check_dynamic(x) for x in val)
            return False

        if check_dynamic(self.blend): 
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")

        base_blend = self.blend
        op_type = self.op_type
        is_linear = self.blend_type == 'linear'
        
        if op_type == 'morph':
            base_t = np.clip(base_blend, 0.0, 1.0)
            mask_callable = self.mask.to_callable() if self.mask else None
            falloff = max(self.mask_falloff, 1e-4)

            def _callable_morph(p):
                res = child_callables[0](p)
                t = base_t
                if mask_callable:
                    d_mask = mask_callable(p)
                    factor = 1.0 - _smoothstep(0.0, falloff, d_mask)
                    t = t * factor
                
                for i in range(1, len(child_callables)):
                    d_next = child_callables[i](p)
                    res = (1.0 - t) * res + t * d_next
                return res
            return _callable_morph

        if base_blend <= 1e-6 and not self.mask:
            if op_type == 'union':
                return lambda p: reduce(np.minimum, [c(p) for c in child_callables])
            elif op_type == 'intersection':
                return lambda p: reduce(np.maximum, [c(p) for c in child_callables])
            elif op_type == 'difference':
                def _diff_hard(p):
                    res = child_callables[0](p)
                    for c in child_callables[1:]:
                        res = np.maximum(res, -c(p))
                    return res
                return _diff_hard

        mask_callable = self.mask.to_callable() if self.mask else None
        falloff = max(self.mask_falloff, 1e-4)

        def _callable_smooth(points: np.ndarray) -> np.ndarray:
            dists = [c(points) for c in child_callables]
            blend = base_blend
            if mask_callable:
                d_mask = mask_callable(points)
                factor = 1.0 - _smoothstep(0.0, falloff, d_mask)
                blend = blend * factor
            
            blend = np.maximum(blend, 1e-6)

            res = dists[0]

            for i in range(1, len(dists)):
                d1 = res
                d2 = dists[i]

                if op_type == 'union':
                    h = np.clip(0.5 + 0.5 * (d2 - d1) / blend, 0.0, 1.0)
                    res = d2 * (1.0 - h) + d1 * h
                    if not is_linear: res -= blend * h * (1.0 - h)
                elif op_type == 'intersection':
                    h = np.clip(0.5 - 0.5 * (d2 - d1) / blend, 0.0, 1.0)
                    res = d2 * (1.0 - h) + d1 * h
                    if not is_linear: res += blend * h * (1.0 - h)
                elif op_type == 'difference':
                    h = np.clip(0.5 - 0.5 * (d2 + d1) / blend, 0.0, 1.0)
                    res = d1 * (1.0 - h) + (-d2) * h
                    if not is_linear: res += blend * h * (1.0 - h)
            return res

        return _callable_smooth

    def to_callable(self):
        return self._make_callable([c.to_callable() for c in self.children])

    def to_profile_callable(self):
        return self._make_callable([c.to_profile_callable() for c in self.children])

    def _collect_materials(self, materials: list):
        for c in self.children:
            c._collect_materials(materials)
        if self.mask:
            self.mask._collect_materials(materials)

def Group(*children):
    """
    Creates a union of multiple SDF objects.
    Acts as a helper factory for Compositor.
    """
    return Compositor(list(children), op_type='union')