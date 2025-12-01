import numpy as np
import time
import struct
import sys
from pathlib import Path
from functools import lru_cache
from skimage import measure

GLSL_SOURCES = {}
GLSL_ORDER = ['utils', 'operators', 'compositors', 'primitives', 'scene']
IMPLICIT_DEPENDENCIES = {}

def load_all_glsl():
    if GLSL_SOURCES: return
    glsl_dir = Path(__file__).parent.parent / 'glsl'
    if not glsl_dir.exists(): 
        return
    for glsl_file in glsl_dir.glob('*.glsl'):
        with open(glsl_file, 'r') as f: GLSL_SOURCES[glsl_file.stem] = f.read()

@lru_cache(maxsize=None)
def get_glsl_definitions(required_files: frozenset) -> str:
    if not GLSL_SOURCES: load_all_glsl()
    expanded_files = set(required_files)
    for req in required_files:
        if req in IMPLICIT_DEPENDENCIES: expanded_files.update(IMPLICIT_DEPENDENCIES[req])
    def sort_key(name):
        try: return GLSL_ORDER.index(name)
        except ValueError: return len(GLSL_ORDER) + 1
    sorted_files = sorted(list(expanded_files), key=sort_key)
    return "\n\n".join([GLSL_SOURCES[stem] for stem in sorted_files if stem in GLSL_SOURCES])

def _get_glsl_from_lib(rel_path: str) -> str:
    try:
        glsl_path = Path(__file__).parent.parent / 'glsl' / rel_path
        with open(glsl_path, 'r') as f: return f.read()
    except FileNotFoundError: return ""

def assemble_standalone_shader(sdf_obj) -> str:
    from .scene import SceneCompiler
    materials, uniforms, params = [], {}, {}
    sdf_obj._collect_materials(materials)
    sdf_obj._collect_uniforms(uniforms)
    sdf_obj._collect_params(params)
    all_user_uniforms = list(uniforms.keys()) + [p.uniform_name for p in params.values()]
    custom_uniforms_glsl = "\n".join([f"uniform float {name};" for name in all_user_uniforms])
    material_struct_glsl = "struct MaterialInfo { vec3 color; };\n"
    material_uniform_glsl = f"uniform MaterialInfo u_materials[{max(1, len(materials))}];\n"
    material_lookup_glsl = "int material_id = int(hit.y); vec3 material_color = vec3(0.8); if (material_id >= 0 && material_id < {count}) {{ material_color = u_materials[material_id].color; }}".format(count=len(materials))
    scene_code = SceneCompiler().compile(sdf_obj)
    
    return f"""
#version 330 core
uniform vec2 u_resolution; uniform float u_time;
uniform vec3 u_cam_pos = vec3(5.0, 4.0, 5.0); uniform vec3 u_cam_target = vec3(0.0, 0.0, 0.0); uniform float u_cam_zoom = 1.0;
uniform vec3 u_light_pos = vec3(4.0, 5.0, 6.0); uniform float u_ambient_strength = 0.1; uniform float u_shadow_softness = 8.0; uniform float u_ao_strength = 3.0;
{custom_uniforms_glsl}
{material_struct_glsl} {material_uniform_glsl}
out vec4 f_color;
vec4 Scene(in vec3 p);
{_get_glsl_from_lib('utils.glsl')}
{_get_glsl_from_lib('scene.glsl')}
{scene_code}
void main() {{
    vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
    vec3 ro, rd; cameraStatic(st, u_cam_pos, u_cam_target, u_cam_zoom, ro, rd);
    vec4 hit = raymarch(ro, rd); float t = hit.x; vec3 color = vec3(0.1, 0.12, 0.15);
    if (t > 0.0) {{
        vec3 p = ro + t * rd; vec3 normal = estimateNormal(p);
        vec3 lightDir = normalize(u_light_pos - p);
        float diffuse = max(dot(normal, lightDir), u_ambient_strength);
        float shadow = softShadow(p + normal * 0.01, lightDir, u_shadow_softness);
        float ao = ambientOcclusion(p, normal, u_ao_strength);
        {material_lookup_glsl}
        color = material_color * diffuse * shadow * ao;
    }}
    f_color = vec4(color, 1.0);
}}
"""

def _cartesian_product(*arrays):
    la = len(arrays); dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)): arr[...,i] = a
    return arr.reshape(-la, la)

def _write_binary_stl(path, points):
    n = len(points); points = np.array(points, dtype='float32')
    normals = np.cross(points[:,1] - points[:,0], points[:,2] - points[:,0])
    norm = np.linalg.norm(normals, axis=1).reshape((-1, 1))
    normals /= np.where(norm == 0, 1, norm)
    dtype = np.dtype([('normal', ('<f', 3)), ('points', ('<f', (3, 3))), ('attr', '<H')])
    a = np.zeros(n, dtype=dtype); a['points'] = points; a['normal'] = normals
    with open(path, 'wb') as fp: fp.write(b'\x00' * 80); fp.write(struct.pack('<I', n)); fp.write(a.tobytes())

def _write_obj(path, verts, faces):
    with open(path, 'w') as fp:
        for v in verts: fp.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces + 1: fp.write(f"f {f[0]} {f[1]} {f[2]}\n")

def _write_glb(path, verts, faces, vertex_colors):
    try: import pygltflib
    except ImportError: print("ERROR: Exporting to .glb requires 'pygltflib'. pip install pygltflib", file=sys.stderr); return
    verts_binary = verts.astype('f4').tobytes(); faces_binary = faces.astype('u2').tobytes()
    buffer_data = verts_binary + faces_binary
    gltf = pygltflib.GLTF2(); gltf.scenes.append(pygltflib.Scene(nodes=[0])); gltf.nodes.append(pygltflib.Node(mesh=0))
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(buffer_data)))
    gltf.bufferViews.extend([
        pygltflib.BufferView(buffer=0, byteOffset=0, byteLength=len(verts_binary), target=pygltflib.ARRAY_BUFFER),
        pygltflib.BufferView(buffer=0, byteOffset=len(verts_binary), byteLength=len(faces_binary), target=pygltflib.ELEMENT_ARRAY_BUFFER)
    ])
    gltf.accessors.extend([
        pygltflib.Accessor(bufferView=0, componentType=pygltflib.FLOAT, count=len(verts), type=pygltflib.VEC3, min=np.min(verts, axis=0).tolist(), max=np.max(verts, axis=0).tolist()),
        pygltflib.Accessor(bufferView=1, componentType=pygltflib.UNSIGNED_SHORT, count=len(faces.ravel()), type=pygltflib.SCALAR)
    ])
    gltf.meshes.append(pygltflib.Mesh(primitives=[pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=0), indices=1)]))
    gltf.set_binary_blob(buffer_data); gltf.save(path)

def _adaptive_meshing(sdf_callable, bounds, max_depth, verbose):
    if verbose: print(f"  - Using adaptive meshing with max depth {max_depth}.")
    root_min, root_max = np.array(bounds[0]), np.array(bounds[1])
    voxels = np.array([[root_min, root_max]])
    point_cache = {}
    for depth in range(max_depth):
        if verbose: print(f"  - Octree depth {depth+1}/{max_depth}, evaluating {len(voxels)} voxels...")
        mins, maxs = voxels[:, 0], voxels[:, 1]
        centers = (mins + maxs) / 2.0
        all_corners = np.array([mins, maxs, np.stack([mins[:,0], mins[:,1], maxs[:,2]], -1), np.stack([mins[:,0], maxs[:,1], mins[:,2]], -1), np.stack([maxs[:,0], mins[:,1], mins[:,2]], -1), np.stack([maxs[:,0], maxs[:,1], mins[:,2]], -1), np.stack([maxs[:,0], mins[:,1], maxs[:,2]], -1), np.stack([mins[:,0], maxs[:,1], maxs[:,2]], -1)]).transpose(1,0,2)
        points_to_eval_set = set()
        for p in np.concatenate([all_corners.reshape(-1, 3), centers]):
            if tuple(p) not in point_cache: points_to_eval_set.add(tuple(p))
        if points_to_eval_set:
            dists = sdf_callable(np.array(list(points_to_eval_set), dtype='f4'))
            for i, p in enumerate(points_to_eval_set): point_cache[p] = dists[i]
        surface_mask = np.zeros(len(voxels), dtype=bool)
        voxel_diagonals = np.linalg.norm(maxs - mins, axis=1)
        for i in range(len(voxels)):
            corner_dists = [point_cache[tuple(c)] for c in all_corners[i]]
            center_dist = point_cache[tuple(centers[i])]
            if min(corner_dists) * max(corner_dists) <= 0 or abs(center_dist) < voxel_diagonals[i] * 0.866:
                surface_mask[i] = True
        surface_voxels = voxels[surface_mask]
        if len(surface_voxels) == 0: return [], {}, 0
        if depth < max_depth - 1:
            mins, maxs = surface_voxels[:, 0], surface_voxels[:, 1]
            centers = (mins + maxs) / 2.0
            mx, my, mz, cx, cy, cz, Mx, My, Mz = mins[:,0], mins[:,1], mins[:,2], centers[:,0], centers[:,1], centers[:,2], maxs[:,0], maxs[:,1], maxs[:,2]
            children = [
                (np.stack([mx, my, mz], 1), np.stack([cx, cy, cz], 1)), (np.stack([cx, my, mz], 1), np.stack([Mx, cy, cz], 1)),
                (np.stack([mx, cy, mz], 1), np.stack([cx, My, cz], 1)), (np.stack([mx, my, cz], 1), np.stack([cx, cy, Mz], 1)),
                (np.stack([cx, cy, mz], 1), np.stack([Mx, My, cz], 1)), (np.stack([cx, my, cz], 1), np.stack([Mx, cy, Mz], 1)),
                (np.stack([mx, cy, cz], 1), np.stack([cx, My, Mz], 1)), (np.stack([cx, cy, cz], 1), np.stack([Mx, My, Mz], 1))
            ]
            voxels = np.concatenate([np.stack([m, M], axis=1) for m, M in children], axis=0)
        else: voxels = surface_voxels
    grid_coords = np.array(list(point_cache.keys()))
    min_coord, step = np.min(grid_coords, axis=0), (voxels[0, 1, :] - voxels[0, 0, :])[0]
    indices = np.round((grid_coords - min_coord) / (step + 1e-9)).astype(int)
    max_indices = np.max(indices, axis=0)
    volume = np.full(max_indices + 1, np.linalg.norm(root_max - root_min), dtype='f4')
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = list(point_cache.values())
    return volume, min_coord, step

def _dual_contouring(sdf_callable, bounds, resolution, verbose):
    if verbose: print(f"  - Using Dual Contouring with resolution {resolution}.")
    X, Y, Z = np.linspace(bounds[0][0], bounds[1][0], resolution[0]), np.linspace(bounds[0][1], bounds[1][1], resolution[1]), np.linspace(bounds[0][2], bounds[1][2], resolution[2])
    Nx, Ny, Nz = len(X), len(Y), len(Z)
    volume = sdf_callable(_cartesian_product(X, Y, Z).astype('f4')).reshape(Nx, Ny, Nz)
    def get_normals(points):
        eps = 1e-5; p = np.array(points, dtype='f4')
        dx = sdf_callable(p + np.array([eps, 0, 0])) - sdf_callable(p - np.array([eps, 0, 0]))
        dy = sdf_callable(p + np.array([0, eps, 0])) - sdf_callable(p - np.array([0, eps, 0]))
        dz = sdf_callable(p + np.array([0, 0, eps])) - sdf_callable(p - np.array([0, 0, eps]))
        n = np.stack([dx, dy, dz], axis=-1); nm = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / np.where(nm == 0, 1, nm)
    cell_to_vert, verts, faces = {}, [], []
    corners = np.array([[i,j,k] for i in (0,1) for j in (0,1) for k in (0,1)])
    edges = [[0,1], [2,3], [4,5], [6,7], [0,2], [1,3], [4,6], [5,7], [0,4], [1,5], [2,6], [3,7]]
    if verbose: print("  - Finding feature points...")
    for i in range(Nx-1):
        for j in range(Ny-1):
            for k in range(Nz-1):
                vals = volume[i:i+2, j:j+2, k:k+2].flatten()
                if np.all(vals > 0) or np.all(vals < 0): continue
                cross_pts = []
                for v0, v1 in edges:
                    idx0, idx1 = corners[v0]+[i,j,k], corners[v1]+[i,j,k]
                    val0, val1 = volume[tuple(idx0)], volume[tuple(idx1)]
                    if val0 * val1 < 0:
                        p0, p1 = np.array([X[idx0[0]], Y[idx0[1]], Z[idx0[2]]]), np.array([X[idx1[0]], Y[idx1[1]], Z[idx1[2]]])
                        cross_pts.append(p0 + (val0/(val0-val1)) * (p1-p0))
                if not cross_pts: continue
                cross_norms = get_normals(np.array(cross_pts))
                A = np.sum([np.outer(n, n) for n in cross_norms], axis=0)
                b = np.sum([np.dot(n, p) * n for n, p in zip(cross_norms, cross_pts)], axis=0)
                try: vert = np.clip(np.linalg.pinv(A) @ b, [X[i], Y[j], Z[k]], [X[i+1], Y[j+1], Z[k+1]])
                except np.linalg.LinAlgError: vert = np.mean(cross_pts, axis=0)
                cell_to_vert[(i,j,k)] = len(verts); verts.append(vert)
    if verbose: print("  - Generating faces...")
    for i in range(Nx-1):
        for j in range(Ny-1):
            for k in range(Nz-1):
                def get_v(ix, iy, iz): return cell_to_vert.get((ix, iy, iz))
                if j>0 and k>0 and volume[i,j,k]*volume[i+1,j,k]<0:
                    v = [get_v(i,j-1,k-1), get_v(i,j,k-1), get_v(i,j,k), get_v(i,j-1,k)]
                    if all(x is not None for x in v): faces.extend([[v[0],v[1],v[2]], [v[0],v[2],v[3]]] if volume[i,j,k]<0 else [[v[0],v[3],v[2]], [v[0],v[2],v[1]]])
                if i>0 and k>0 and volume[i,j,k]*volume[i,j+1,k]<0:
                    v = [get_v(i-1,j,k-1), get_v(i,j,k-1), get_v(i,j,k), get_v(i-1,j,k)]
                    if all(x is not None for x in v): faces.extend([[v[0],v[3],v[2]], [v[0],v[2],v[1]]] if volume[i,j,k]<0 else [[v[0],v[1],v[2]], [v[0],v[2],v[3]]])
                if i>0 and j>0 and volume[i,j,k]*volume[i,j,k+1]<0:
                    v = [get_v(i-1,j-1,k), get_v(i,j-1,k), get_v(i,j,k), get_v(i-1,j,k)]
                    if all(x is not None for x in v): faces.extend([[v[0],v[1],v[2]], [v[0],v[2],v[3]]] if volume[i,j,k]<0 else [[v[0],v[3],v[2]], [v[0],v[2],v[1]]])
    return np.array(verts), np.array(faces)

def generate(sdf_obj, bounds=None, samples=2**22, algorithm='marching_cubes', adaptive=False, octree_depth=8, decimate_ratio=None, voxel_size=None, verbose=True):
    if algorithm not in ['marching_cubes', 'dual_contouring']: algorithm = 'marching_cubes'
    if bounds is None: bounds = sdf_obj.estimate_bounds(verbose=verbose)
    min_c, max_c = np.array(bounds[0]), np.array(bounds[1])
    if voxel_size:
        if adaptive: octree_depth = max(2, int(np.ceil(np.log2(np.max(max_c-min_c)/voxel_size)))) if np.max(max_c-min_c)>1e-9 else 1
    if verbose: print(f"INFO: Generating mesh... Bounds: {bounds}")
    try: sdf_callable = sdf_obj.to_callable()
    except Exception as e: raise RuntimeError(f"Could not generate mesh: {e}")
    
    verts, faces = [], []
    if adaptive:
        if algorithm == 'dual_contouring': raise ValueError("Dual Contouring does not support adaptive meshing.")
        volume, origin, step = _adaptive_meshing(sdf_callable, bounds, octree_depth, verbose)
        if len(volume):
            try: verts, faces, _, _ = measure.marching_cubes(volume, level=0, spacing=(step, step, step)); verts += origin
            except: pass
    elif algorithm == 'dual_contouring':
        res = tuple(max(4, int((max_c[i]-min_c[i])/voxel_size)) for i in range(3)) if voxel_size else tuple([max(4, int(samples**(1/3)))]*3)
        verts, faces = _dual_contouring(sdf_callable, bounds, res, verbose)
    else:
        step = voxel_size if voxel_size else ((max_c[0]-min_c[0])*(max_c[1]-min_c[1])*(max_c[2]-min_c[2])/samples)**(1/3)
        X, Y, Z = np.arange(min_c[0], max_c[0], step), np.arange(min_c[1], max_c[1], step), np.arange(min_c[2], max_c[2], step)
        dist = sdf_callable(_cartesian_product(X, Y, Z).astype('f4')).reshape(len(X), len(Y), len(Z))
        try: verts, faces, _, _ = measure.marching_cubes(dist, level=0, spacing=(step, step, step)); verts += min_c
        except: pass
        
    if len(verts)==0: print("ERROR: Mesh generation failed.", file=sys.stderr); return np.array([]), np.array([])
    if decimate_ratio and 0 < decimate_ratio < 1:
        try:
            import trimesh
            if verbose: print(f"  - Simplifying mesh...")
            mesh = trimesh.Trimesh(vertices=verts, faces=faces).simplify_quadric_decimation(int(len(faces)*(1.0-decimate_ratio)))
            verts, faces = mesh.vertices, mesh.faces
        except ImportError: print("WARNING: 'trimesh' required for simplification.", file=sys.stderr)
    return verts, faces

def save(sdf_obj, path, bounds, samples, verbose, algorithm, adaptive, vertex_colors, decimate_ratio=None, octree_depth=8, voxel_size=None):
    start = time.time()
    verts, faces = generate(sdf_obj, bounds, samples, algorithm, adaptive, octree_depth, decimate_ratio, voxel_size, verbose)
    if len(verts) == 0: return
    if verbose: print(f"INFO: Saving to '{path}'...")
    if path.lower().endswith('.stl'): _write_binary_stl(path, verts[faces])
    elif path.lower().endswith('.obj'): _write_obj(path, verts, faces)
    elif path.lower().endswith(('.glb', '.gltf')): 
        if vertex_colors: print("WARNING: vertex_colors not implemented for GLB.", file=sys.stderr)
        _write_glb(path, verts, faces, vertex_colors)
    else: print(f"ERROR: Unsupported format '{path}'", file=sys.stderr); return
    if verbose: print(f"SUCCESS: Saved {len(faces)} triangles in {time.time()-start:.2f}s.")