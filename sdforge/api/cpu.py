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

# --- Noise Helpers ---
def _mod289(x): return x - np.floor(x * (1.0 / 289.0)) * 289.0
def _permute(x): return _mod289(((x*34.0)+1.0)*x)
def _taylorInvSqrt(r): return 1.79284291400159 - 0.85373472095314 * r

def _snoise3(v):
    C = np.array([1.0/6.0, 1.0/3.0])
    D = np.array([0.0, 0.5, 1.0, 2.0])

    # First corner
    dot_v_Cyyy = np.sum(v, axis=1) * C[1]
    i = np.floor(v + dot_v_Cyyy[:, None])
    dot_i_Cxxx = np.sum(i, axis=1) * C[0]
    x0 = v - i + dot_i_Cxxx[:, None]

    # Other corners
    g = (x0[:, [1, 2, 0]] < x0).astype(np.float32)
    l = 1.0 - g
    i1 = np.minimum(g, l[:, [2, 0, 1]])
    i2 = np.maximum(g, l[:, [2, 0, 1]])

    x1 = x0 - i1 + C[0]
    x2 = x0 - i2 + C[1]
    x3 = x0 - D[1]

    # Permutations
    i = _mod289(i)
    ix, iy, iz = i[:,0][:,None], i[:,1][:,None], i[:,2][:,None]
    i1x, i1y, i1z = i1[:,0][:,None], i1[:,1][:,None], i1[:,2][:,None]
    i2x, i2y, i2z = i2[:,0][:,None], i2[:,1][:,None], i2[:,2][:,None]

    term_z = iz + np.hstack([np.zeros_like(iz), i1z, i2z, np.ones_like(iz)])
    p = _permute(term_z)
    term_y = iy + np.hstack([np.zeros_like(iy), i1y, i2y, np.ones_like(iy)])
    p = _permute(p + term_y)
    term_x = ix + np.hstack([np.zeros_like(ix), i1x, i2x, np.ones_like(ix)])
    p = _permute(p + term_x)

    # Gradients
    ns = 0.142857142857
    j = p - 49.0 * np.floor(p * ns * ns)
    x_ = np.floor(j * ns)
    y_ = np.floor(j - 7.0 * x_)
    x = x_ * ns + ns
    y = y_ * ns + ns
    h = 1.0 - np.abs(x) - np.abs(y)

    b0 = np.stack([x[:,0], x[:,1], y[:,0], y[:,1]], axis=1)
    b1 = np.stack([x[:,2], x[:,3], y[:,2], y[:,3]], axis=1)
    s0 = np.floor(b0) * 2.0 + 1.0
    s1 = np.floor(b1) * 2.0 + 1.0
    sh = -np.where(h < 0, 1.0, 0.0)

    b0_xzyw = b0[:, [0, 2, 1, 3]]
    s0_xzyw = s0[:, [0, 2, 1, 3]]
    sh_xxyy = sh[:, [0, 0, 1, 1]]
    a0 = b0_xzyw + s0_xzyw * sh_xxyy

    b1_xzyw = b1[:, [0, 2, 1, 3]]
    s1_xzyw = s1[:, [0, 2, 1, 3]]
    sh_zzww = sh[:, [2, 2, 3, 3]]
    a1 = b1_xzyw + s1_xzyw * sh_zzww

    p0 = np.stack([a0[:,0], a0[:,1], h[:,0]], axis=1)
    p1 = np.stack([a0[:,2], a0[:,3], h[:,1]], axis=1)
    p2 = np.stack([a1[:,0], a1[:,1], h[:,2]], axis=1)
    p3 = np.stack([a1[:,2], a1[:,3], h[:,3]], axis=1)

    norm = _taylorInvSqrt(np.stack([np.sum(p0*p0, axis=1), np.sum(p1*p1, axis=1), np.sum(p2*p2, axis=1), np.sum(p3*p3, axis=1)], axis=1))
    p0 *= norm[:, 0:1]; p1 *= norm[:, 1:2]; p2 *= norm[:, 2:3]; p3 *= norm[:, 3:4]

    m = np.maximum(0.6 - np.stack([np.sum(x0*x0, axis=1), np.sum(x1*x1, axis=1), np.sum(x2*x2, axis=1), np.sum(x3*x3, axis=1)], axis=1), 0.0)
    m2 = m * m; m4 = m2 * m2
    gradients = np.stack([np.sum(p0*x0, axis=1), np.sum(p1*x1, axis=1), np.sum(p2*x2, axis=1), np.sum(p3*x3, axis=1)], axis=1)
    return 42.0 * np.sum(m4 * gradients, axis=1)

def _snoiseVec3(p):
    s = _snoise3(p)
    p1 = np.stack([p[:,1] - 19.1, p[:,2] + 33.4, p[:,0] + 47.2], axis=1)
    s1 = _snoise3(p1)
    p2 = np.stack([p[:,2] + 74.2, p[:,0] - 124.5, p[:,1] + 99.4], axis=1)
    s2 = _snoise3(p2)
    return np.stack([s, s1, s2], axis=1)

# --- Bezier Helper ---
def _solve_cubic(a, b, c):
    # Vectorized solver for t^3 + a*t^2 + b*t + c = 0
    p = b - a*a / 3.0
    q = 2.0*a*a*a / 27.0 - a*b / 3.0 + c
    p3 = p*p*p
    d = q*q / 4.0 + p3 / 27.0
    
    mask_d_pos = d > 0
    roots = np.zeros((len(a), 3))
    
    # 1 Real Root
    if np.any(mask_d_pos):
        d_sqrt = np.sqrt(d[mask_d_pos])
        q_val = q[mask_d_pos]
        a_val = a[mask_d_pos]
        u = np.cbrt(-q_val / 2.0 + d_sqrt)
        v = np.cbrt(-q_val / 2.0 - d_sqrt)
        t0 = u + v - a_val / 3.0
        roots[mask_d_pos] = t0[:, None]

    # 3 Real Roots
    mask_d_neg = ~mask_d_pos
    if np.any(mask_d_neg):
        p_val = p[mask_d_neg]
        q_val = q[mask_d_neg]
        a_val = a[mask_d_neg]
        r = np.sqrt(-p_val / 3.0)
        phi = np.arccos(np.clip(-q_val / (2.0 * r * r * r), -1.0, 1.0))
        t1 = 2.0 * r * np.cos(phi / 3.0) - a_val / 3.0
        t2 = 2.0 * r * np.cos((phi + 2.0*np.pi) / 3.0) - a_val / 3.0
        t3 = 2.0 * r * np.cos((phi + 4.0*np.pi) / 3.0) - a_val / 3.0
        roots[mask_d_neg, 0] = t1
        roots[mask_d_neg, 1] = t2
        roots[mask_d_neg, 2] = t3
    return roots

# --- 2D SDF Implementation Helpers ---
def _sd_circle(p, r):
    return np.linalg.norm(p, axis=1) - r

def _sd_rectangle(p, b):
    d = np.abs(p) - b
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(np.maximum(d[:,0], d[:,1]), 0.0)

def _sd_triangle(p, r):
    k = np.sqrt(3.0)
    p = p.copy()
    p[:,0] = np.abs(p[:,0]) - r
    p[:,1] = p[:,1] + r/k
    
    mask = p[:,0] + k*p[:,1] > 0.0
    if np.any(mask):
        px, py = p[mask, 0], p[mask, 1]
        p[mask, 0] = (px - k*py)/2.0
        p[mask, 1] = (-k*px - py)/2.0
        
    p[:,0] -= np.clip(p[:,0], -2.0*r, 0.0)
    return -np.linalg.norm(p, axis=1) * np.sign(p[:,1])

def _sd_trapezoid(p, r1, r2, he):
    k1 = np.array([r2, he])
    k2 = np.array([r2 - r1, 2.0 * he])
    p = p.copy()
    p[:,0] = np.abs(p[:,0])
    ca = np.stack([p[:,0] - np.minimum(p[:,0], np.where(p[:,1] < 0, r1, r2)), np.abs(p[:,1]) - he], axis=1)
    
    cb = p - k1 + k2 * np.clip(np.sum((k1 - p) * k2, axis=1, keepdims=True) / np.dot(k2, k2), 0.0, 1.0)
    s = np.where((cb[:,0] < 0.0) & (ca[:,1] < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(np.sum(ca*ca, axis=1), np.sum(cb*cb, axis=1)))

# --- Primitives ---
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

def _line(node, profile_mode):
    _validate_no_params(node, ['radius'])
    a, b, r = node.start, node.end, node.radius
    ba = b - a
    baba = np.dot(ba, ba)
    
    if profile_mode:
        # Flatten to 2D (XY plane)
        a = np.array([a[0], a[1], 0.0])
        b = np.array([b[0], b[1], 0.0])
        ba = b - a
        baba = np.dot(ba, ba)

    def func(p):
        p_in = p if not profile_mode else np.stack([p[:,0], p[:,1], np.zeros(len(p))], axis=1)
        pa = p_in - a
        h = np.clip(np.sum(pa * ba, axis=-1) / baba, 0.0, 1.0)
        return np.linalg.norm(pa - ba * h[:, np.newaxis], axis=-1) - r
    return func

def _cone(node):
    _validate_no_params(node, ['height', 'radius_base', 'radius_top'])
    h, r1, r2 = node.height, node.radius_base, node.radius_top
    def func(p):
        q = np.stack([np.linalg.norm(p[:, [0, 2]], axis=1), p[:, 1]], axis=1)
        if r2 < 1e-6: # Cone
            w = np.array([r1, h])
            a = q - w * np.clip(np.sum(q * w, axis=1, keepdims=True) / np.dot(w, w), 0.0, 1.0)
            b = q - np.stack([np.zeros(len(q)), np.clip(q[:, 1], 0.0, h)], axis=1)
            k = np.sign(r1)
            d = np.minimum(np.sum(a*a, axis=1), np.sum(b*b, axis=1))
            s = np.maximum(k*(q[:,0]*w.y - q[:,1]*w.x), k*(q[:,1] - h)) # w.y is h, w.x is r1
            return np.sqrt(d) * np.sign(s)
        else: # Capped Cone
            k1 = np.array([r2, h])
            k2 = np.array([r2 - r1, 2.0 * h])
            ca = np.stack([q[:,0] - np.minimum(q[:,0], np.where(q[:,1] < 0, r1, r2)), np.abs(q[:,1]) - h], axis=1)
            cb = q - k1 + k2 * np.clip(np.sum((k1 - q) * k2, axis=1, keepdims=True) / np.dot(k2, k2), 0.0, 1.0)
            s = np.where((cb[:,0] < 0.0) & (ca[:,1] < 0.0), -1.0, 1.0)
            return s * np.sqrt(np.minimum(np.sum(ca*ca, axis=1), np.sum(cb*cb, axis=1)))
    return func

def _hex_prism(node):
    _validate_no_params(node, ['radius', 'height'])
    r, h = node.radius, node.height / 2.0
    k = np.array([-0.8660254, 0.5, 0.57735026])
    def func(p):
        p_abs = np.abs(p)
        dot_k_p = p_abs[:,0]*k[0] + p_abs[:,1]*k[1]
        p_xy = p_abs[:,:2] - 2.0 * np.minimum(dot_k_p, 0.0)[:,None] * k[:2]
        
        d1 = np.linalg.norm(p_xy - np.stack([np.clip(p_xy[:,0], -k[2]*r, k[2]*r), np.full(len(p), r)], axis=1), axis=1) * np.sign(p_xy[:,1] - r)
        d2 = p_abs[:,2] - h
        return np.minimum(np.maximum(d1, d2), 0.0) + np.linalg.norm(np.maximum(np.stack([d1, d2], axis=1), 0.0), axis=1)
    return func

def _octahedron(node):
    _validate_no_params(node, ['size'])
    s = node.size
    def func(p):
        p_abs = np.abs(p)
        return (p_abs[:,0] + p_abs[:,1] + p_abs[:,2] - s) * 0.57735027
    return func

def _ellipsoid(node):
    _validate_no_params(node, ['radii'])
    r = np.array(node.radii)
    def func(p):
        k0 = np.linalg.norm(p / r, axis=1)
        k1 = np.linalg.norm(p / (r * r), axis=1)
        return k0 * (k0 - 1.0) / k1
    return func

def _pyramid(node):
    _validate_no_params(node, ['height'])
    h = node.height
    def func(p):
        p_adj = p.copy()
        p_adj[:,1] += h * 0.5
        m2 = h*h + 0.25
        p_xz = np.abs(p_adj[:,[0,2]])
        swap_mask = p_xz[:,1] > p_xz[:,0]
        p_xz[swap_mask] = p_xz[swap_mask][:, ::-1]
        p_xz -= 0.5
        
        q = np.stack([p_xz[:,1], h*p_adj[:,1] - 0.5*p_xz[:,0], h*p_xz[:,0] + 0.5*p_adj[:,1]], axis=1)
        s = np.maximum(-q[:,0], 0.0)
        t = np.clip((q[:,1] - 0.5*p_xz[:,1]) / (m2 + 0.25), 0.0, 1.0)
        
        a = m2 * (q[:,0]+s)**2 + q[:,1]**2
        b = m2 * (q[:,0]+0.5*t)**2 + (q[:,1]-m2*t)**2
        
        d2 = np.where(np.minimum(q[:,1], -q[:,0]*m2 - q[:,1]*0.5) > 0.0, 0.0, np.minimum(a, b))
        return np.sqrt((d2 + q[:,2]**2) / m2) * np.sign(np.maximum(q[:,2], -p_adj[:,1]))
    return func

def _bezier(node, profile_mode):
    _validate_no_params(node, ['radius'])
    A, B, C, r = node.p0, node.p1, node.p2, node.radius
    if profile_mode:
        A = np.array([A[0], A[1], 0.0])
        B = np.array([B[0], B[1], 0.0])
        C = np.array([C[0], C[1], 0.0])
        
    def func(p):
        p_in = p if not profile_mode else np.stack([p[:,0], p[:,1], np.zeros(len(p))], axis=1)
        a = B - A
        b = A - 2.0*B + C
        c = a * 2.0
        d = A - p_in
        
        kk = 1.0 / np.dot(b, b)
        kx = kk * np.dot(a, b)
        
        ddb = np.einsum('ij,j->i', d, b)
        ky = kk * (2.0 * np.dot(a, a) + ddb) / 3.0
        
        dda = np.einsum('ij,j->i', d, a)
        kz = kk * dda
        
        # solve cubic for t
        roots = _solve_cubic(np.full(len(p), 3.0*kx), 3.0*ky, kz)
        roots = np.clip(roots, 0.0, 1.0)
        
        # Evaluate distance squared at each root
        d2 = np.full(len(p), np.inf)
        for i in range(3):
            t = roots[:, i]
            # q = d + (c + b*t)*t
            # broadcast b (3,) with t (N,) -> (N, 3)
            bt = b[None, :] * t[:, None]
            c_plus_bt = c + bt
            term = c_plus_bt * t[:, None]
            q = d + term
            d2 = np.minimum(d2, np.sum(q*q, axis=1))
            
        return np.sqrt(d2) - r
    return func

def _polyline(node, profile_mode):
    # Polyline iterates over segments and takes the min distance
    _validate_no_params(node, ['radius'])
    points = np.array(node.points)
    r = node.radius
    segments = []
    
    n_seg = len(points) if node.closed else len(points) - 1
    for i in range(n_seg):
        p0 = points[i]
        p1 = points[(i + 1) % len(points)]
        
        if profile_mode:
             p0 = np.array([p0[0], p0[1], 0.0])
             p1 = np.array([p1[0], p1[1], 0.0])
             
        segments.append((p0, p1))
        
    def func(p):
        p_in = p if not profile_mode else np.stack([p[:,0], p[:,1], np.zeros(len(p))], axis=1)
        d_min = np.full(len(p), np.inf)
        
        for p0, p1 in segments:
            ba = p1 - p0
            pa = p_in - p0
            h = np.clip(np.sum(pa * ba, axis=-1) / np.dot(ba, ba), 0.0, 1.0)
            dist = np.linalg.norm(pa - ba * h[:, np.newaxis], axis=-1) - r
            d_min = np.minimum(d_min, dist)
            
        return d_min
    return func

def _polycurve(node, profile_mode):
    # Polycurve iterates over segments (Line or Bezier) and takes min
    _validate_no_params(node, ['radius'])
    segments_data = node._generate_segments()
    r = node.radius
    
    # Pre-process segments
    processed_segs = []
    for A, B, C in segments_data:
        if profile_mode:
            A = np.array([A[0], A[1], 0.0])
            B = np.array([B[0], B[1], 0.0])
            C = np.array([C[0], C[1], 0.0])
        
        is_line = np.allclose(B, C) or np.allclose(A, B)
        processed_segs.append((A, B, C, is_line))
        
    def func(p):
        p_in = p if not profile_mode else np.stack([p[:,0], p[:,1], np.zeros(len(p))], axis=1)
        d_min = np.full(len(p), np.inf)
        
        for A, B, C, is_line in processed_segs:
            if is_line:
                # sdCapsule logic (A to C)
                ba = C - A
                pa = p_in - A
                h = np.clip(np.sum(pa * ba, axis=-1) / np.dot(ba, ba), 0.0, 1.0)
                dist = np.linalg.norm(pa - ba * h[:, np.newaxis], axis=-1) - r
            else:
                # sdBezier logic
                a = B - A
                b = A - 2.0*B + C
                c = a * 2.0
                d = A - p_in
                
                kk = 1.0 / np.dot(b, b)
                kx = kk * np.dot(a, b)
                
                ddb = np.einsum('ij,j->i', d, b)
                ky = kk * (2.0 * np.dot(a, a) + ddb) / 3.0
                dda = np.einsum('ij,j->i', d, a)
                kz = kk * dda
                
                roots = _solve_cubic(np.full(len(p), 3.0*kx), 3.0*ky, kz)
                roots = np.clip(roots, 0.0, 1.0)
                
                d2 = np.full(len(p), np.inf)
                for i in range(3):
                    t = roots[:, i]
                    bt = b[None, :] * t[:, None]
                    term = (c + bt) * t[:, None]
                    q = d + term
                    d2 = np.minimum(d2, np.sum(q*q, axis=1))
                dist = np.sqrt(d2) - r
                
            d_min = np.minimum(d_min, dist)
        return d_min
    return func

def _wrap_2d_primitive(node, func_2d, profile_mode):
    def func(p):
        if profile_mode:
            return func_2d(p[:, :2])
        else:
            d_2d = func_2d(p[:, :2])
            # Thin plate logic: max(d_2d, abs(z) - thickness)
            # Default thickness 0.001
            w = np.stack([d_2d, np.abs(p[:, 2]) - 0.001], axis=1)
            return np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=1)
    return func

def _circle(node, profile_mode):
    _validate_no_params(node, ['radius'])
    r = node.radius
    return _wrap_2d_primitive(node, lambda p2: _sd_circle(p2, r), profile_mode)

def _rectangle(node, profile_mode):
    _validate_no_params(node, ['size'])
    b = np.array(node.size) / 2.0
    return _wrap_2d_primitive(node, lambda p2: _sd_rectangle(p2, b), profile_mode)

def _triangle(node, profile_mode):
    _validate_no_params(node, ['radius'])
    r = node.radius
    return _wrap_2d_primitive(node, lambda p2: _sd_triangle(p2, r), profile_mode)

def _trapezoid(node, profile_mode):
    _validate_no_params(node, ['bottom_width', 'top_width', 'height'])
    r1, r2, he = node.bottom_width / 2.0, node.top_width / 2.0, node.height / 2.0
    return _wrap_2d_primitive(node, lambda p2: _sd_trapezoid(p2, r1, r2, he), profile_mode)

# --- Operations ---

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

def _morph(node, child_fns):
    factor = node.factor
    fn_a, fn_b = child_fns[0], child_fns[1]
    def func(p):
        d1 = fn_a(p)
        d2 = fn_b(p)
        return d1 + (d2 - d1) * factor
    return func

# --- Transforms ---

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
    # Applying inverse rotation (transpose)
    return lambda p: child_fn(p @ mat.T)

def _orient(node, child_fn):
    axis = node.axis
    def func(p):
        p_new = p.copy()
        if np.allclose(axis, [1,0,0]): # X -> Y, Y -> Z, Z -> X (p.zyx)
            p_new = p[:, [2, 1, 0]]
        elif np.allclose(axis, [0,1,0]): # Y -> X, X -> Z, Z -> Y (p.xzy)
            p_new = p[:, [0, 2, 1]]
        return child_fn(p_new)
    return func

def _twist(node, child_fn):
    k = node.strength
    def func(p):
        theta = -k * p[:, 1]
        c = np.cos(theta)
        s = np.sin(theta)
        x, z = p[:, 0], p[:, 2]
        p_new = p.copy()
        p_new[:, 0] = c * x + s * z
        p_new[:, 2] = -s * x + c * z
        return child_fn(p_new)
    return func

def _bend(node, child_fn):
    axis, k = node.axis, node.curvature
    def func(p):
        p_new = p.copy()
        if np.allclose(axis, [1,0,0]): # Bend X
            c, s = np.cos(k * p[:,0]), np.sin(k * p[:,0])
            p_new[:,1] = c * p[:,1] + s * p[:,2]
            p_new[:,2] = -s * p[:,1] + c * p[:,2]
        elif np.allclose(axis, [0,1,0]): # Bend Y
            c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
            p_new[:,0] = c * p[:,0] - s * p[:,2]
            p_new[:,2] = s * p[:,0] + c * p[:,2]
        else: # Bend Z
            c, s = np.cos(k * p[:,2]), np.sin(k * p[:,2])
            p_new[:,0] = c * p[:,0] + s * p[:,1]
            p_new[:,1] = -s * p[:,0] + c * p[:,1]
        return child_fn(p_new)
    return func

def _warp(node, child_fn):
    freq, strength = node.frequency, node.strength
    def func(p):
        offset = _snoiseVec3(p * freq) * strength
        return child_fn(p + offset)
    return func

def _repeat(node, child_fn):
    spacing, count, axis = node.spacing, node.count, node.axis
    def func(p):
        if spacing is not None and count is not None: # Limited
            c_vec = np.ones(3)*count if np.size(count)==1 else count
            lim = np.round(p / spacing)
            lim = np.clip(lim, -c_vec, c_vec)
            return child_fn(p - spacing * lim)
        elif spacing is not None: # Infinite
            return child_fn(np.mod(p + 0.5 * spacing, spacing) - 0.5 * spacing)
        else: # Polar
            c = count if np.size(count)==1 else count[0]
            angle = 2.0 * np.pi / c
            # Only supports Y-axis for now to match GLSL mostly
            # Convert to polar
            if np.allclose(axis, [1,0,0]): # X-axis
                r = np.linalg.norm(p[:,[1,2]], axis=1)
                a = np.arctan2(p[:,1], p[:,2])
                new_a = np.mod(a + 0.5 * angle, angle) - 0.5 * angle
                p_new = np.stack([p[:,0], r * np.sin(new_a), r * np.cos(new_a)], axis=1)
            elif np.allclose(axis, [0,0,1]): # Z-axis
                r = np.linalg.norm(p[:,[0,1]], axis=1)
                a = np.arctan2(p[:,0], p[:,1])
                new_a = np.mod(a + 0.5 * angle, angle) - 0.5 * angle
                p_new = np.stack([r * np.sin(new_a), r * np.cos(new_a), p[:,2]], axis=1)
            else: # Y-axis (default)
                r = np.linalg.norm(p[:,[0,2]], axis=1)
                a = np.arctan2(p[:,0], p[:,2])
                new_a = np.mod(a + 0.5 * angle, angle) - 0.5 * angle
                p_new = np.stack([r * np.sin(new_a), p[:,1], r * np.cos(new_a)], axis=1)
            return child_fn(p_new)
    return func

def _mirror(node, child_fn):
    axes = node.axes
    def func(p):
        p_new = p.copy()
        if axes[0] > 0.5: p_new[:,0] = np.abs(p_new[:,0])
        if axes[1] > 0.5: p_new[:,1] = np.abs(p_new[:,1])
        if axes[2] > 0.5: p_new[:,2] = np.abs(p_new[:,2])
        return child_fn(p_new)
    return func

# --- Shaping ---

def _round(node, child_fn):
    r = node.radius
    return lambda p: child_fn(p) - r

def _shell(node, child_fn):
    th = node.thickness
    return lambda p: np.abs(child_fn(p)) - th

def _extrude(node, child_fn):
    h = node.height
    def func(p):
        # We need the 2D profile distance. 
        # Since child_fn was created with profile_mode=True, 
        # it treats p as (x,y,0) inside, or effectively 2D.
        # But wait, child_fn for 2D primitives already takes (N,3) or (N,2) depending on implementation.
        # Our implementation of 2D primitives in profile_mode takes (N,3) and ignores Z, returning 2D distance.
        # Perfect.
        
        d = child_fn(p) # This is d_2d
        
        # Extrude logic: w = vec2(d, abs(p.z) - h)
        w = np.stack([d, np.abs(p[:,2]) - h], axis=1)
        return np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=1)
    return func

def _revolve(node, child_fn):
    def func(p):
        # Revolve around Y axis. Profile defined in XY plane with X > 0.
        # q = vec2(length(p.xz), p.y)
        # We pass vec3(q.x, q.y, 0) to child profile function
        q_x = np.linalg.norm(p[:, [0, 2]], axis=1)
        q_y = p[:, 1]
        p_profile = np.stack([q_x, q_y, np.zeros(len(p))], axis=1)
        return child_fn(p_profile)
    return func

def _displace_by_noise(node, child_fn):
    scale, strength = node.scale, node.strength
    def func(p):
        d = child_fn(p)
        n = _snoise3(p * scale) * strength
        return d + n
    return func

# --- Dispatch ---

def get_callable(node, profile_mode=False):
    node_type = type(node).__name__
    
    # Recursive helper to pass profile_mode correctly
    def recurse(c, mode=profile_mode):
        return get_callable(c, profile_mode=mode)

    child_fns = []
    if hasattr(node, 'children'):
        child_fns = [recurse(c) for c in node.children]
    elif hasattr(node, 'child') and node.child:
        if node_type in ['Extrude', 'Revolve']:
            # Force profile mode for children of shapers
            child_fns = [recurse(node.child, mode=True)]
        else:
            child_fns = [recurse(node.child)]
    elif hasattr(node, 'a') and hasattr(node, 'b'):
        child_fns = [recurse(node.a), recurse(node.b)]

    if node_type == 'Sphere': return _sphere(node)
    if node_type == 'Box': return _box(node)
    if node_type == 'Cylinder': return _cylinder(node)
    if node_type == 'Plane': return _plane(node)
    if node_type == 'Torus': return _torus(node)
    
    # Primitives that support profile mode
    if node_type == 'Line': return _line(node, profile_mode)
    if node_type == 'Polyline': return _polyline(node, profile_mode)
    if node_type == 'Polycurve': return _polycurve(node, profile_mode)
    if node_type == 'Bezier': return _bezier(node, profile_mode)
    if node_type == 'Circle': return _circle(node, profile_mode)
    if node_type == 'Rectangle': return _rectangle(node, profile_mode)
    if node_type == 'Triangle': return _triangle(node, profile_mode)
    if node_type == 'Trapezoid': return _trapezoid(node, profile_mode)

    if node_type == 'Cone': return _cone(node)
    if node_type == 'HexPrism': return _hex_prism(node)
    if node_type == 'Octahedron': return _octahedron(node)
    if node_type == 'Ellipsoid': return _ellipsoid(node)
    if node_type == 'Pyramid': return _pyramid(node)

    if node_type == 'Group': return _union(node, child_fns)
    if node_type == 'Union': return _union(node, child_fns)
    if node_type == 'Difference': return _difference(node, child_fns)
    if node_type == 'Intersection': return _intersection(node, child_fns)
    if node_type == 'Morph': return _morph(node, child_fns)
    
    if node_type == 'Translate': return _translate(node, child_fns[0])
    if node_type == 'Scale': return _scale(node, child_fns[0])
    if node_type == 'Rotate': return _rotate(node, child_fns[0])
    if node_type == 'Orient': return _orient(node, child_fns[0])
    if node_type == 'Twist': return _twist(node, child_fns[0])
    if node_type == 'Bend': return _bend(node, child_fns[0])
    if node_type == 'Warp': return _warp(node, child_fns[0])
    if node_type == 'Repeat': return _repeat(node, child_fns[0])
    if node_type == 'Mirror': return _mirror(node, child_fns[0])
    
    if node_type == 'Round': return _round(node, child_fns[0])
    if node_type == 'Shell': return _shell(node, child_fns[0])
    if node_type == 'Extrude': return _extrude(node, child_fns[0])
    if node_type == 'Revolve': return _revolve(node, child_fns[0])
    if node_type == 'DisplaceByNoise': return _displace_by_noise(node, child_fns[0])

    if node_type in ['Forge', 'Displace']:
        raise NotImplementedError(f"Node type '{node_type}' using raw GLSL strings is not supported by the CPU backend. Use backend='gpu'.")
    
    raise NotImplementedError(f"CPU implementation for '{node_type}' is not currently available.")