import numpy as np
from functools import reduce

def _validate_no_params(obj, attrs):
    from .params import Param
    for attr in attrs:
        val = getattr(obj, attr, None)
        if isinstance(val, (str, Param)):
            raise TypeError(f"CPU Backend does not support interactive Params for '{attr}'. Use backend='gpu'.")
        if isinstance(val, (list, tuple, np.ndarray)):
            for v in val:
                if isinstance(v, (str, Param)):
                    raise TypeError(f"CPU Backend does not support interactive Params in '{attr}'.")

def _sphere(node):
    _validate_no_params(node, ['radius'])
    r = node.radius
    return lambda p: np.linalg.norm(p, axis=-1) - r

def _box(node):
    _validate_no_params(node, ['size'])
    half_size = np.array(node.size) / 2.0
    def func(p):
        q = np.abs(p) - half_size
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)
    return func

def _cylinder(node):
    _validate_no_params(node, ['radius', 'height'])
    r, h = node.radius, node.height / 2.0
    def func(p):
        d = np.abs(np.array([np.linalg.norm(p[:, [0, 2]], axis=-1), p[:, 1]]).T) - np.array([r, h])
        return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    return func

def _plane(node):
    _validate_no_params(node, ['offset'])
    n, o = node.normal, node.offset
    return lambda p: np.dot(p, n) + o

def _torus(node):
    _validate_no_params(node, ['radius_major', 'radius_minor'])
    maj, min_r = node.radius_major, node.radius_minor
    def func(p):
        q = np.array([np.linalg.norm(p[:, [0, 2]], axis=-1) - maj, p[:, 1]]).T
        return np.linalg.norm(q, axis=-1) - min_r
    return func

def _line(node):
    _validate_no_params(node, ['radius'])
    a, b, r = node.start, node.end, node.radius
    ba = b - a
    baba = np.dot(ba, ba)
    def func(p):
        pa = p - a
        h = np.clip(np.sum(pa * ba, axis=-1) / baba, 0.0, 1.0)
        return np.linalg.norm(pa - ba * h[:, np.newaxis], axis=-1) - r
    return func

def _union(node, child_fns):
    if node.blend > 1e-6:
        k = node.blend
        def func(p):
            dists = [f(p) for f in child_fns]
            res = dists[0]
            for i in range(1, len(dists)):
                d1, d2 = res, dists[i]
                h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
                res = d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)
            return res
        return func
    return lambda p: reduce(np.minimum, [f(p) for f in child_fns])

def _difference(node, child_fns):
    fn_a, fn_b = child_fns[0], child_fns[1]
    if node.blend > 1e-6:
        k = node.blend
        def func(p):
            d1, d2 = fn_a(p), -fn_b(p)
            h = np.clip(0.5 - 0.5 * (d1 - d2) / k, 0.0, 1.0)
            return d1 * (1.0 - h) + d2 * h + k * h * (1.0 - h)
        return func
    return lambda p: np.maximum(fn_a(p), -fn_b(p))

def _intersection(node, child_fns):
    if node.blend > 1e-6:
        k = node.blend
        def func(p):
            dists = [f(p) for f in child_fns]
            res = dists[0]
            for i in range(1, len(dists)):
                d1, d2 = res, dists[i]
                h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
                res = d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h)
            return res
        return func
    return lambda p: reduce(np.maximum, [f(p) for f in child_fns])

def _translate(node, child_fn):
    off = node.offset
    return lambda p: child_fn(p - off)

def _scale(node, child_fn):
    f = node.factor
    correction = np.mean(f)
    return lambda p: child_fn(p / f) * correction

def _rotate(node, child_fn):
    axis, angle = node.axis, node.angle
    c, s = np.cos(angle), np.sin(angle)
    if np.allclose(axis, [1,0,0]): mat = np.array([[1,0,0],[0,c,s],[0,-s,c]])
    elif np.allclose(axis, [0,1,0]): mat = np.array([[c,0,-s],[0,1,0],[s,0,c]])
    elif np.allclose(axis, [0,0,1]): mat = np.array([[c,s,0],[-s,c,0],[0,0,1]])
    else:
        kx, ky, kz = axis
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        mat = np.eye(3) + s * K + (1 - c) * (K @ K)
    return lambda p: child_fn(p @ mat.T)

def _round(node, child_fn):
    r = node.radius
    return lambda p: child_fn(p) - r

def _shell(node, child_fn):
    th = node.thickness
    return lambda p: np.abs(child_fn(p)) - th

def get_callable(node):
    node_type = type(node).__name__
    
    child_fns = []
    if hasattr(node, 'children'):
        child_fns = [get_callable(c) for c in node.children]
    elif hasattr(node, 'child') and node.child:
        child_fns = [get_callable(node.child)]
    elif hasattr(node, 'a') and hasattr(node, 'b'):
        child_fns = [get_callable(node.a), get_callable(node.b)]

    if node_type == 'Sphere': return _sphere(node)
    if node_type == 'Box': return _box(node)
    if node_type == 'Cylinder': return _cylinder(node)
    if node_type == 'Plane': return _plane(node)
    if node_type == 'Torus': return _torus(node)
    if node_type == 'Line': return _line(node)

    if node_type == 'Group': return _union(node, child_fns)
    if node_type == 'Union': return _union(node, child_fns)
    if node_type == 'Difference': return _difference(node, child_fns)
    if node_type == 'Intersection': return _intersection(node, child_fns)
    
    if node_type == 'Translate': return _translate(node, child_fns[0])
    if node_type == 'Scale': return _scale(node, child_fns[0])
    if node_type == 'Rotate': return _rotate(node, child_fns[0])
    
    if node_type == 'Round': return _round(node, child_fns[0])
    if node_type == 'Shell': return _shell(node, child_fns[0])

    if node_type in ['Forge', 'Displace', 'DisplaceByNoise', 'Warp', 'Twist', 'Bend']:
        raise NotImplementedError(f"Node type '{node_type}' is not supported by the CPU backend. Use backend='gpu'.")
    
    raise NotImplementedError(f"CPU implementation for '{node_type}' is not currently available.")