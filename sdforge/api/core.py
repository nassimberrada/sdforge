import numpy as np
from abc import ABC, abstractmethod
import sys

X, Y, Z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])

class GLSLContext:
    """Manages the state of the GLSL compilation process for a scene."""
    def __init__(self, compiler):
        self.compiler = compiler
        self.p = "p"
        self.statements = []
        self.dependencies = set()
        self._var_counter = 0
        self.definitions = set()

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
        new_ctx.dependencies = self.dependencies.copy()
        new_ctx.definitions = self.definitions.copy()
        new_ctx._var_counter = self._var_counter
        return new_ctx

    def merge_from(self, sub_context: 'GLSLContext'):
        """Merges statements and state from a sub-context into this one."""
        self.statements.extend(sub_context.statements)
        self.dependencies.update(sub_context.dependencies)
        self.definitions.update(sub_context.definitions)
        self._var_counter = sub_context._var_counter


class SDFNode(ABC):
    """Abstract base class for all SDF objects in the scene graph."""

    glsl_dependencies = set()

    def __init__(self):
        super().__init__()
        if not hasattr(self, 'child'):
            self.child = None
        self.mask = None

    def _collect_params(self, params: dict):
        """Recursively collects Param objects from the scene graph."""
        from .utils import Param, Expr

        def check(val):
            if isinstance(val, Param):
                params[val.uniform_name] = val
            elif isinstance(val, Expr):
                for p in val.params:
                    params[p.uniform_name] = p

        for attr_name in dir(self):
            if attr_name.startswith('_') or attr_name in ['child', 'children', 'mask']:
                continue
            try:
                attr_val = getattr(self, attr_name)
                check(attr_val)
                if isinstance(attr_val, (list, tuple, np.ndarray)):
                    for item in attr_val:
                        check(item)
            except Exception:
                continue

        if hasattr(self, 'child') and self.child:
            self.child._collect_params(params)
        if hasattr(self, 'children'):
            for child in self.children:
                child._collect_params(params)
        if hasattr(self, 'mask') and self.mask:
            self.mask._collect_params(params)

    def _collect_materials(self, materials: list):
        """Recursively collects Material objects from the scene graph."""
        if hasattr(self, 'child') and self.child:
            self.child._collect_materials(materials)
        if hasattr(self, 'children'):
            for child in self.children:
                child._collect_materials(materials)
        if hasattr(self, 'mask') and self.mask:
            self.mask._collect_materials(materials)

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

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        """Returns the GLSL for the 2D profile of this object (ignoring Z)."""
        return self.to_glsl(ctx)

    def to_profile_callable(self):
        """Returns a callable for the 2D profile of this object."""
        return self.to_callable()

    def render(self, camera=None, light=None, debug=None, mode='auto', **kwargs):
        """Renders the SDF object."""
        from .scene import render as render_func
        return render_func(self, camera=camera, light=light, debug=debug, mode=mode, **kwargs)

    def save(self, path, bounds=None, samples=2**22, verbose=True, algorithm='marching_cubes', adaptive=False, octree_depth=8, vertex_colors=False, decimate_ratio=None, voxel_size=None):
        """Generates a mesh and saves it to a file."""
        if bounds is None:
            if verbose:
                print("INFO: No bounds provided to .save(), estimating automatically.", file=sys.stderr)
            bounds = self.estimate_bounds(verbose=verbose)
        from .io import save as save_func
        save_func(self, path, bounds, samples, verbose, algorithm, adaptive, vertex_colors, decimate_ratio, octree_depth=octree_depth, voxel_size=voxel_size)

    def save_frame(self, path, camera=None, light=None, **kwargs):
        """Renders a single frame and saves it to an image file."""
        self.render(save_frame=path, watch=False, camera=camera, light=light, **kwargs)

    def estimate_bounds(self, resolution=64, search_bounds=((-5, -5, -5), (5, 5, 5)), padding=0.1, verbose=True):
        """Estimates the bounding box of the SDF object by sampling a grid."""
        from .io import _cartesian_product
        if verbose:
            print(f"INFO: Estimating bounds with {resolution**3} samples...", file=sys.stderr)

        sdf_callable = self.to_callable()
        axes = [np.linspace(search_bounds[0][i], search_bounds[1][i], resolution) for i in range(3)]
        points_grid = _cartesian_product(*axes).astype('f4')

        distances = sdf_callable(points_grid)
        inside_mask = distances <= 1e-4
        inside_points = points_grid[inside_mask]

        if inside_points.shape[0] < 2:
            if verbose:
                print(f"WARNING: No object surface found within the search bounds {search_bounds}.", file=sys.stderr)
            return search_bounds

        min_coords = np.min(inside_points, axis=0)
        max_coords = np.max(inside_points, axis=0)
        step_size = np.array([(search_bounds[1][i] - search_bounds[0][i]) / (resolution - 1) for i in range(3)])
        min_coords -= step_size
        max_coords += step_size
        size = max_coords - min_coords
        size[size < 1e-6] = padding
        min_coords -= size * padding
        max_coords += size * padding
        return (tuple(min_coords), tuple(max_coords))

    def export_shader(self, path: str):
        """Exports a complete, self-contained GLSL fragment shader."""
        from .io import assemble_standalone_shader
        shader_code = assemble_standalone_shader(self)
        with open(path, 'w') as f:
            f.write(shader_code)
        print(f"SUCCESS: Shader exported to '{path}'.")

    def _collect_uniforms(self, uniforms: dict):
        if hasattr(self, 'child') and self.child:
            self.child._collect_uniforms(uniforms)
        if hasattr(self, 'children'):
            for child in self.children:
                child._collect_uniforms(uniforms)
        if hasattr(self, 'mask') and self.mask:
            self.mask._collect_uniforms(uniforms)

    def union(self, *others, blend: float = 0.0, blend_type: str = 'smooth', mask: 'SDFNode' = None, mask_falloff: float = 0.0) -> 'SDFNode':
        from .compositors import Compositor
        return Compositor(children=[self] + list(others), op_type='union', blend=blend, blend_type=blend_type, mask=mask, mask_falloff=mask_falloff)

    def intersection(self, *others, blend: float = 0.0, blend_type: str = 'smooth', mask: 'SDFNode' = None, mask_falloff: float = 0.0) -> 'SDFNode':
        from .compositors import Compositor
        return Compositor(children=[self] + list(others), op_type='intersection', blend=blend, blend_type=blend_type, mask=mask, mask_falloff=mask_falloff)

    def difference(self, other, blend: float = 0.0, blend_type: str = 'smooth', mask: 'SDFNode' = None, mask_falloff: float = 0.0) -> 'SDFNode':
        from .compositors import Compositor
        return Compositor(children=[self, other], op_type='difference', blend=blend, blend_type=blend_type, mask=mask, mask_falloff=mask_falloff)

    def morph(self, other, factor: float = 0.5, mask: 'SDFNode' = None, mask_falloff: float = 0.0) -> 'SDFNode':
        from .compositors import Compositor
        return Compositor(children=[self, other], op_type='morph', blend=factor, mask=mask, mask_falloff=mask_falloff)

    def __or__(self, other): return self.union(other)
    def __and__(self, other): return self.intersection(other)
    def __sub__(self, other): return self.difference(other)

    def align_to(self, reference_point, face_normal, offset: float = 0.0) -> 'SDFNode':
        from .operators import align_to_face
        return align_to_face(self, reference_point, face_normal, offset)

    def place_at_angle(self, pivot_point, axis, angle_rad, distance) -> 'SDFNode':
        from .operators import place_at_angle
        return place_at_angle(self, pivot_point, axis, angle_rad, distance)

    def offset_along(self, reference_point, direction, distance) -> 'SDFNode':
        from .operators import offset_along
        return offset_along(self, reference_point, direction, distance)

    def bounding_box(self, padding: float = 0.0) -> 'SDFNode':
        from .operators import bounding_box
        return bounding_box(self, padding)

    def stack(self, other: 'SDFNode', direction, spacing: float = 0.0) -> 'SDFNode':
        from .operators import stack
        return stack(self, other, direction, spacing)

    def color(self, r, g=None, b=None, mask: 'SDFNode' = None) -> 'SDFNode':
        from .operators import Operator
        if isinstance(r, (list, tuple, np.ndarray)):
            rgb = tuple(r)
        else:
            rgb = (r, g, b)
        return Operator(self, op_type='material', func_name='', params=[rgb], mask=mask)

    def translate(self, offset, mask=None, mask_falloff=0.0) -> 'SDFNode':
        from .operators import Operator
        from .utils import Param

        off = np.array(offset)
        has_params = any(isinstance(x, Param) for x in off.flatten())
        
        inv_func = None if has_params else lambda p: p - off
        return Operator(self, 'transform', "opTranslate", [off], inverse_func=inv_func, mask=mask, mask_falloff=mask_falloff)

    def scale(self, factor, mask=None, mask_falloff=0.0) -> 'SDFNode':
        from .operators import Operator
        from .utils import Param

        if isinstance(factor, (int, float, str, Param)):
            f = np.array([factor, factor, factor])
            corr = factor
        else:
            f = np.array(factor)
            corr = (f[0] + f[1] + f[2]) / 3.0

        has_params = any(isinstance(x, Param) for x in f.flatten())
        
        inv_func = None if has_params else lambda p: p / f
        return Operator(self, 'transform', "opScale", [f], inverse_func=inv_func, dist_correction=corr, mask=mask, mask_falloff=mask_falloff)

    def rotate(self, axis, angle: float, mask=None, mask_falloff=0.0) -> 'SDFNode':
        from .operators import Operator
        from .utils import Param

        ax = np.array(axis, dtype=float)
        if np.linalg.norm(ax) == 0: raise ValueError("Rotation axis cannot be zero vector")
        ax /= np.linalg.norm(ax)

        has_params = isinstance(angle, Param)
        if has_params:
            inv_func = None
            if np.allclose(ax, X): 
                return Operator(self, 'transform', "opRotateX", [angle], inverse_func=None, mask=mask, mask_falloff=mask_falloff)
            elif np.allclose(ax, Y):
                return Operator(self, 'transform', "opRotateY", [angle], inverse_func=None, mask=mask, mask_falloff=mask_falloff)
            elif np.allclose(ax, Z):
                return Operator(self, 'transform', "opRotateZ", [angle], inverse_func=None, mask=mask, mask_falloff=mask_falloff)
            else:
                return Operator(self, 'transform', "opRotateAxis", [ax, angle], inverse_func=None, mask=mask, mask_falloff=mask_falloff)

        if np.allclose(ax, X): 
            func_name = "opRotateX"
            params = [angle]
            inv_func = lambda p: p @ np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        elif np.allclose(ax, Y):
            func_name = "opRotateY"
            params = [angle]
            inv_func = lambda p: p @ np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        elif np.allclose(ax, Z):
            func_name = "opRotateZ"
            params = [angle]
            inv_func = lambda p: p @ np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        else:
            func_name = "opRotateAxis"
            params = [ax, angle]
            def inv_func(p):
                c, s = np.cos(angle), np.sin(angle)
                kx, ky, kz = ax
                K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
                rot = np.eye(3) + s * K + (1 - c) * (K @ K)
                return p @ rot 

        return Operator(self, 'transform', func_name, params, inverse_func=inv_func, mask=mask, mask_falloff=mask_falloff)

    def orient(self, axis) -> 'SDFNode':
        from .operators import Operator
        axis_map = {'x': X, 'y': Y, 'z': Z}
        if isinstance(axis, str) and axis.lower() in axis_map:
            ax = axis_map[axis.lower()]
        else:
            ax = axis
        if np.allclose(ax, X): return Operator(self, 'transform', "opOrientX", [], inverse_func=lambda p: p[:, [2,1,0]])
        elif np.allclose(ax, Y): return Operator(self, 'transform', "opOrientY", [], inverse_func=lambda p: p[:, [0,2,1]])
        return self

    def twist(self, strength: float, mask=None, mask_falloff=0.0) -> 'SDFNode':
        from .operators import Operator
        from .utils import Param
        if isinstance(strength, Param): inv_func = None
        else:
            def inv_func(p):
                c, s = np.cos(strength * p[:,1]), np.sin(strength * p[:,1])
                x_new = p[:,0]*c - p[:,2]*s
                z_new = p[:,0]*s + p[:,2]*c
                return np.stack([x_new, p[:,1], z_new], axis=-1)
        return Operator(self, 'transform', "opTwist", [strength], inverse_func=inv_func, mask=mask, mask_falloff=mask_falloff)

    def bend(self, axis, curvature: float, mask=None, mask_falloff=0.0) -> 'SDFNode':
        from .operators import Operator
        from .utils import Param
        k = curvature
        has_params = isinstance(k, Param)
        if has_params:
            inv_func = None
        else:
            if np.allclose(axis, X):
                def inv_func(p):
                    c, s = np.cos(k * p[:,0]), np.sin(k * p[:,0])
                    return np.stack([p[:,0], p[:,1]*c - p[:,2]*s, p[:,1]*s + p[:,2]*c], axis=-1)
            elif np.allclose(axis, Y):
                def inv_func(p):
                    c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
                    return np.stack([p[:,0]*c - p[:,2]*s, p[:,1], p[:,0]*s + p[:,2]*c], axis=-1)
            else:
                def inv_func(p):
                    c, s = np.cos(k * p[:,2]), np.sin(k * p[:,2])
                    return np.stack([p[:,0]*c - p[:,1]*s, p[:,0]*s + p[:,1]*c, p[:,2]], axis=-1)

        if np.allclose(axis, X): fname = "opBendX"
        elif np.allclose(axis, Y): fname = "opBendY"
        else: fname = "opBendZ"
        return Operator(self, 'transform', fname, [curvature], inverse_func=inv_func, mask=mask, mask_falloff=mask_falloff)

    def warp(self, frequency: float, strength: float, mask=None, mask_falloff=0.0) -> 'SDFNode':
        from .operators import Operator
        return Operator(self, 'transform', "opWarp", [frequency, strength], inverse_func=None, mask=mask, mask_falloff=mask_falloff)

    def repeat(self, spacing) -> 'SDFNode':
        from .operators import Operator
        from .utils import Param
        s = np.array(spacing)
        has_params = any(isinstance(x, Param) for x in s.flatten())
        
        if has_params: inv_func = None
        else:
            def inv_func(p):
                q = p.copy()
                m = s != 0
                q[:, m] = np.mod(p[:, m] + 0.5 * s[m], s[m]) - 0.5 * s[m]
                return q
        return Operator(self, 'transform', "opRepeat", [s], inverse_func=inv_func)

    def limited_repeat(self, spacing, limits) -> 'SDFNode':
        from .operators import Operator
        from .utils import Param
        s = np.array(spacing)
        l = np.array(limits)
        has_params = any(isinstance(x, Param) for x in np.concatenate([s.flatten(), l.flatten()]))
        
        if has_params: inv_func = None
        else:
            def inv_func(p):
                q = p.copy()
                m = s != 0
                rounded = np.round(p[:, m] / (s[m] + 1e-9))
                q[:, m] = p[:, m] - s[m] * np.clip(rounded, -l[m], l[m])
                return q
        return Operator(self, 'transform', "opLimitedRepeat", [s, l], inverse_func=inv_func)

    def polar_repeat(self, repetitions: int) -> 'SDFNode':
        from .operators import Operator
        from .utils import Param
        if isinstance(repetitions, Param): inv_func = None
        else:
            def inv_func(p):
                a = np.arctan2(p[:,0], p[:,2])
                r = np.linalg.norm(p[:,[0,2]], axis=-1)
                angle = 2 * np.pi / repetitions
                newA = np.mod(a, angle) - 0.5 * angle
                return np.stack([r * np.sin(newA), p[:,1], r * np.cos(newA)], axis=-1)
        return Operator(self, 'transform', "opPolarRepeat", [repetitions], inverse_func=inv_func)

    def mirror(self, axes) -> 'SDFNode':
        from .operators import Operator
        from .utils import _glsl_format
        a = np.array(axes)
        
        def inv_func(p):
            q = p.copy()
            if a[0] > 0.5: q[:,0] = np.abs(q[:,0])
            if a[1] > 0.5: q[:,1] = np.abs(q[:,1])
            if a[2] > 0.5: q[:,2] = np.abs(q[:,2])
            return q
        return Operator(self, 'transform', "opMirror", [a], inverse_func=inv_func)

    def __add__(self, offset): return self.translate(offset)
    def __mul__(self, factor): return self.scale(factor)
    def __rmul__(self, factor): return self.scale(factor)

    def round(self, radius: float, mask: 'SDFNode' = None, mask_falloff: float = 0.0) -> 'SDFNode':
        from .operators import Operator
        return Operator(self, 'modify', "opRound", [radius], forward_func=lambda d, r: d - r, mask=mask, mask_falloff=mask_falloff)

    def shell(self, thickness: float, mask: 'SDFNode' = None, mask_falloff: float = 0.0) -> 'SDFNode':
        from .operators import Operator
        return Operator(self, 'modify', "opShell", [thickness], forward_func=lambda d, t: np.abs(d) - t, mask=mask, mask_falloff=mask_falloff)

    def extrude(self, height: float) -> 'SDFNode':
        from .operators import Operator
        return Operator(self, 'extrude', "opExtrude", [height])

    def revolve(self) -> 'SDFNode':
        from .operators import Operator
        return Operator(self, 'revolve', "opRevolve", [])

    def displace(self, displacement_glsl: str, mask=None, mask_falloff=0.0) -> 'SDFNode':
        from .operators import Operator
        return Operator(self, 'modify', "opDisplace", [displacement_glsl], forward_func=None, mask=mask, mask_falloff=mask_falloff)

    def displace_by_noise(self, scale: float = 10.0, strength: float = 0.1, mask=None, mask_falloff=0.0) -> 'SDFNode':
        from .operators import Operator
        from .utils import _glsl_format
        expr = f"snoise(p * {_glsl_format(scale)}) * {_glsl_format(strength)}"
        return Operator(self, 'modify', "opDisplace", [expr], forward_func=None, mask=mask, mask_falloff=mask_falloff)