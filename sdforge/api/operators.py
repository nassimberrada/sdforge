import numpy as np
from .core import SDFNode, GLSLContext, X, Y, Z
from .utils import _glsl_format, _smoothstep, Param
from .primitives import box, Primitive
from .compositors import Group

class Operator(SDFNode):
    """
    A generic node for operations that alter geometry, space, or surface properties.
    Unifies Transform, Modify, Extrude, Revolve, and Material logic.
    """
    glsl_dependencies = {"operators"}

    def __init__(self, child: SDFNode, op_type: str, func_name: str, params: list, 
                 inverse_func=None, forward_func=None, dist_correction=None, 
                 mask: SDFNode = None, mask_falloff: float = 0.0, material_id=None):
        super().__init__()
        self.child = child
        self.op_type = op_type
        self.func_name = func_name
        self.params = params
        
        self.inverse_func = inverse_func
        self.forward_func = forward_func
        
        self.dist_correction = dist_correction
        self.mask = mask
        self.mask_falloff = mask_falloff
        self.material_id = material_id

        if func_name == 'opWarp' or "noise" in func_name.lower():
            self.glsl_dependencies = {"operators"}

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        
        # --- MATERIAL ---
        if self.op_type == 'material':
            if profile_mode: child_var = self.child.to_profile_glsl(ctx)
            else: child_var = self.child.to_glsl(ctx)
            
            new_id = float(self.material_id)
            if self.mask:
                mask_var = self.mask.to_glsl(ctx)
                selector = f"step({mask_var}.x, 0.0)"
                id_expr = f"mix({child_var}.y, {new_id}, {selector})"
            else:
                id_expr = f"{new_id}"
            return ctx.new_variable('vec4', f"vec4({child_var}.x, {id_expr}, {child_var}.zw)")

        if self.op_type == 'extrude':
            child_var = self.child.to_profile_glsl(ctx)
            h = _glsl_format(self.params[0])
            return ctx.new_variable('vec4', f"opExtrude({child_var}, {ctx.p}, {h})")

        if self.op_type == 'revolve':
            revolved_p_xy = f"vec2(length({ctx.p}.xz), {ctx.p}.y)"
            transformed_p = ctx.new_variable('vec3', f"vec3({revolved_p_xy}, 0.0)")
            sub_ctx = ctx.with_p(transformed_p)
            child_var = self.child.to_profile_glsl(sub_ctx)
            ctx.merge_from(sub_ctx)
            return child_var

        if self.op_type == 'transform':

            param_strs = [_glsl_format(p) for p in self.params]
            raw_transform_expr = f"{self.func_name}({ctx.p}, {', '.join(param_strs)})" if param_strs else f"{self.func_name}({ctx.p})"

            if self.mask:
                mask_var = self.mask.to_glsl(ctx)
                falloff_str = _glsl_format(self.mask_falloff)
                factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
                final_p_expr = f"mix({ctx.p}, {raw_transform_expr}, {factor_expr})"
                transformed_p = ctx.new_variable('vec3', final_p_expr)
            else:
                transformed_p = ctx.new_variable('vec3', raw_transform_expr)
                factor_expr = None

            sub_ctx = ctx.with_p(transformed_p)
            child_var = self.child.to_profile_glsl(sub_ctx) if profile_mode else self.child.to_glsl(sub_ctx)
            ctx.merge_from(sub_ctx)

            if self.dist_correction is not None:
                corr_val = _glsl_format(self.dist_correction)
                correction_expr = f"mix(1.0, {corr_val}, {factor_expr})" if (self.mask and factor_expr) else corr_val
                return ctx.new_variable('vec4', f"vec4({child_var}.x * ({correction_expr}), {child_var}.yzw)")
            return child_var

        if self.op_type == 'modify':
            child_var = self.child.to_profile_glsl(ctx) if profile_mode else self.child.to_glsl(ctx)
            formatted_params = [_glsl_format(p) for p in self.params]

            if self.mask and formatted_params:
                mask_var = self.mask.to_glsl(ctx)
                falloff_str = _glsl_format(self.mask_falloff)
                factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
                formatted_params[0] = f"({formatted_params[0]} * {factor_expr})"

            param_str = ", ".join(formatted_params)
            return ctx.new_variable('vec4', f"{self.func_name}({child_var}, {param_str})")

        raise ValueError(f"Unknown OpType: {self.op_type}")

    def to_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, False)
    def to_profile_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, True)

    def _collect_materials(self, materials: list):
        if self.op_type == 'material':
            wrapper = _MaterialWrapper(self.params[0]) 
            if wrapper not in materials:
                self.material_id = len(materials)
                wrapper.id = self.material_id
                materials.append(wrapper)
            else:
                self.material_id = materials[materials.index(wrapper)].id
        
        self.child._collect_materials(materials)
        if self.mask: self.mask._collect_materials(materials)

    def to_callable(self):
        return self._make_callable(profile=False)

    def to_profile_callable(self):
        return self._make_callable(profile=True)

    def _make_callable(self, profile):
        def check_dynamic(val):
            if isinstance(val, str): return False 
            if isinstance(val, Param): return True
            if hasattr(val, 'params') and val.params: return True 
            if isinstance(val, (list, tuple, np.ndarray)):
                return any(check_dynamic(x) for x in np.array(val, dtype=object).flatten())
            return False

        if any(check_dynamic(p) for p in self.params):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        if self.dist_correction is not None and check_dynamic(self.dist_correction):
            raise TypeError("Cannot save mesh of an object with animated scale.")

        if self.op_type in ['extrude', 'revolve']:
            child_func = self.child.to_profile_callable()
        else:
            child_func = self.child.to_profile_callable() if profile else self.child.to_callable()

        if self.op_type == 'transform':
            if self.inverse_func is None: raise TypeError(f"Transform '{self.func_name}' is GPU-only.")
            inv_func = self.inverse_func
            correction = self.dist_correction if self.dist_correction is not None else 1.0
            
            if self.mask:
                mask_func = self.mask.to_callable()
                falloff = max(self.mask_falloff, 1e-4)
                def _masked_t(p):
                    dm = mask_func(p)
                    mix_f = 1.0 - _smoothstep(0.0, falloff, dm)
                    f_vec = mix_f[:, np.newaxis]
                    p_trans = inv_func(p)
                    p_mixed = p * (1.0 - f_vec) + p_trans * f_vec
                    d = child_func(p_mixed)
                    if correction != 1.0:
                        corr_mixed = 1.0 * (1.0 - mix_f) + correction * mix_f
                        d *= corr_mixed
                    return d
                return _masked_t
            else:
                if correction == 1.0: return lambda p: child_func(inv_func(p))
                return lambda p: child_func(inv_func(p)) * correction

        elif self.op_type == 'modify':
            if self.forward_func is None: raise TypeError(f"Modifier '{self.func_name}' is GPU-only.")
            fwd_func = self.forward_func
            params = self.params
            if self.mask:
                mask_func = self.mask.to_callable()
                falloff = max(self.mask_falloff, 1e-4)
                def _masked_m(p):
                    d_child = child_func(p)
                    dm = mask_func(p)
                    mix_f = 1.0 - _smoothstep(0.0, falloff, dm)
                    p0 = params[0] * mix_f
                    return fwd_func(d_child, p0, *params[1:])
                return _masked_m
            else:
                return lambda p: fwd_func(child_func(p), *params)

        elif self.op_type == 'extrude':
            h = self.params[0]
            def _ext(p):
                d = child_func(p)
                w = np.stack([d, np.abs(p[:, 2]) - h], axis=-1)
                return np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=-1)
            return _ext

        elif self.op_type == 'revolve':
            def _rev(p):
                r = np.linalg.norm(p[:, [0, 2]], axis=-1)
                y = p[:, 1]
                p_2d = np.stack([r, y, np.zeros_like(r)], axis=-1)
                return child_func(p_2d)
            return _rev

        elif self.op_type == 'material':
            return child_func 

        return child_func

class _MaterialWrapper:
    """Helper to check material equality in list."""
    def __init__(self, color): 
        self.rgb = tuple(color)
        self.id = -1
    def __eq__(self, other): return isinstance(other, _MaterialWrapper) and self.rgb == other.rgb


def align_to_face(obj_to_align, reference_point, face_normal: np.ndarray, offset: float = 0.0):
    """Aligns and places an object onto a conceptual face."""
    face_normal = np.array(face_normal) / np.linalg.norm(face_normal)
    y_axis = Y
    if np.allclose(y_axis, face_normal): rotated_obj = obj_to_align
    elif np.allclose(y_axis, -face_normal): rotated_obj = obj_to_align.rotate(X, np.pi)
    else:
        if np.allclose(np.abs(face_normal), X): rotated_obj = obj_to_align.rotate(Z, -np.sign(face_normal[0]) * np.pi / 2)
        elif np.allclose(np.abs(face_normal), Y): rotated_obj = obj_to_align if face_normal[1] > 0 else obj_to_align.rotate(X, np.pi)
        elif np.allclose(np.abs(face_normal), Z): rotated_obj = obj_to_align.rotate(X, np.sign(face_normal[2]) * np.pi / 2)
        else: raise ValueError("align_to_face currently only supports cardinal axis normals (X, Y, Z).")
    
    obj_half_size = getattr(obj_to_align, 'height', 1.0) / 2.0
    translation = reference_point + face_normal * (obj_half_size + offset)
    return rotated_obj.translate(translation)

def place_at_angle(obj_to_place, pivot_point, axis, angle_rad, distance):
    """Places an object at a specific angle and distance from a pivot point."""
    return obj_to_place.translate(X * distance).rotate(axis, angle_rad).translate(pivot_point)

def offset_along(obj_to_place, reference_point, direction, distance):
    """Moves an object from a reference point along a direction vector."""
    normalized_dir = np.array(direction) / np.linalg.norm(direction)
    destination = np.array(reference_point) + normalized_dir * distance
    return obj_to_place.translate(destination)

def compute_stack_transform(obj_fixed, obj_movable, direction, spacing=0.0):
    direction = np.array(direction, dtype=float)
    len_dir = np.linalg.norm(direction)
    if len_dir == 0: raise ValueError("Direction cannot be zero.")
    direction /= len_dir

    b_fixed = obj_fixed.estimate_bounds(verbose=False)
    b_movable = obj_movable.estimate_bounds(verbose=False)
    min_f, max_f = np.array(b_fixed[0]), np.array(b_fixed[1])
    min_m, max_m = np.array(b_movable[0]), np.array(b_movable[1])
    
    T = ((min_f + max_f) / 2.0) - ((min_m + max_m) / 2.0)
    
    axis_idx = np.argmax(np.abs(direction))
    sign = np.sign(direction[axis_idx])
    fixed_face = max_f[axis_idx] if sign > 0 else min_f[axis_idx]
    movable_face = min_m[axis_idx] if sign > 0 else max_m[axis_idx]
    
    aligned_movable_face_pos = movable_face + T[axis_idx]
    target_pos = fixed_face + (sign * spacing)
    T[axis_idx] += target_pos - aligned_movable_face_pos
    return T

def stack(obj_fixed, obj_movable, direction, spacing=0.0):
    """Stacks `obj_movable` onto `obj_fixed` along the given direction."""
    T = compute_stack_transform(obj_fixed, obj_movable, direction, spacing)
    return obj_fixed | obj_movable.translate(T)

def distribute(objects, direction, spacing=0.0):
    """Arranges a list of objects sequentially along a direction."""
    if not objects: return None
    transformed_list = [objects[0]]
    for i in range(1, len(objects)):
        T = compute_stack_transform(transformed_list[-1], objects[i], direction, spacing)
        transformed_list.append(objects[i].translate(T))
    return Group(*transformed_list)

def bounding_box(sdf_obj, padding: float = 0.0):
    """Creates a `box` primitive that encloses a complex SDF object."""
    bounds = sdf_obj.estimate_bounds(verbose=False)
    min_c, max_c = np.array(bounds[0]), np.array(bounds[1])
    size = max_c - min_c + (2 * padding)
    center = (min_c + max_c) / 2.0
    return box(size=tuple(size)).translate(tuple(center))