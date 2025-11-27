import numpy as np
import time
import struct
import sys
from skimage import measure
from collections import defaultdict

def _cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-la, la)

def _write_binary_stl(path, points):
    n = len(points)
    points = np.array(points, dtype='float32')

    normals = np.cross(points[:,1] - points[:,0], points[:,2] - points[:,0])
    norm = np.linalg.norm(normals, axis=1).reshape((-1, 1))
    normals /= np.where(norm == 0, 1, norm)

    dtype = np.dtype([
        ('normal', ('<f', 3)),
        ('points', ('<f', (3, 3))),
        ('attr', '<H'),
    ])

    a = np.zeros(n, dtype=dtype)
    a['points'] = points
    a['normal'] = normals

    with open(path, 'wb') as fp:
        fp.write(b'\x00' * 80)
        fp.write(struct.pack('<I', n))
        fp.write(a.tobytes())

def _write_obj(path, verts, faces):
    with open(path, 'w') as fp:
        for v in verts:
            fp.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces + 1:
            fp.write(f"f {f[0]} {f[1]} {f[2]}\n")

def _write_glb(path, verts, faces, vertex_colors):
    try:
        import pygltflib
    except ImportError:
        print("ERROR: Exporting to .glb requires 'pygltflib'.", file=sys.stderr)
        print("Please install it via: pip install pygltflib", file=sys.stderr)
        return

    # Convert verts and faces to GLB format
    verts_binary = verts.astype('f4').tobytes()
    faces_binary = faces.astype('u2').tobytes()

    buffer_data = verts_binary + faces_binary
    
    gltf = pygltflib.GLTF2()
    gltf.scenes.append(pygltflib.Scene(nodes=[0]))
    gltf.nodes.append(pygltflib.Node(mesh=0))
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(buffer_data)))
    gltf.bufferViews.extend([
        pygltflib.BufferView(buffer=0, byteOffset=0, byteLength=len(verts_binary), target=pygltflib.ARRAY_BUFFER),
        pygltflib.BufferView(buffer=0, byteOffset=len(verts_binary), byteLength=len(faces_binary), target=pygltflib.ELEMENT_ARRAY_BUFFER),
    ])

    min_pos = np.min(verts, axis=0).tolist()
    max_pos = np.max(verts, axis=0).tolist()
    
    gltf.accessors.extend([
        pygltflib.Accessor(bufferView=0, componentType=pygltflib.FLOAT, count=len(verts), type=pygltflib.VEC3, min=min_pos, max=max_pos),
        pygltflib.Accessor(bufferView=1, componentType=pygltflib.UNSIGNED_SHORT, count=len(faces.ravel()), type=pygltflib.SCALAR),
    ])

    primitive = pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=0), indices=1)
    
    gltf.meshes.append(pygltflib.Mesh(primitives=[primitive]))
    
    gltf.set_binary_blob(buffer_data)
    gltf.save(path)


def _adaptive_meshing(sdf_callable, bounds, max_depth, verbose):
    """
    Generates a sparse grid of SDF evaluations using an octree subdivision approach.
    """
    if verbose:
        print(f"  - Using adaptive meshing with max depth {max_depth}.")

    root_min = np.array(bounds[0])
    root_max = np.array(bounds[1])
    voxels = np.array([[root_min, root_max]])
    point_cache = {}
    
    for depth in range(max_depth):
        if verbose:
            print(f"  - Octree depth {depth+1}/{max_depth}, evaluating {len(voxels)} voxels...")

        # 1. Collect ALL unique points needed for this level (corners and centers)
        mins, maxs = voxels[:, 0], voxels[:, 1]
        centers = (mins + maxs) / 2.0
        
        all_corners = np.array([
            mins, maxs, np.stack([mins[:,0], mins[:,1], maxs[:,2]], -1),
            np.stack([mins[:,0], maxs[:,1], mins[:,2]], -1), np.stack([maxs[:,0], mins[:,1], mins[:,2]], -1),
            np.stack([maxs[:,0], maxs[:,1], mins[:,2]], -1), np.stack([maxs[:,0], mins[:,1], maxs[:,2]], -1),
            np.stack([mins[:,0], maxs[:,1], maxs[:,2]], -1)
        ]).transpose(1,0,2)

        points_to_eval_set = set()
        for p in np.concatenate([all_corners.reshape(-1, 3), centers]):
            p_tuple = tuple(p)
            if p_tuple not in point_cache:
                points_to_eval_set.add(p_tuple)

        # 2. Batch evaluate only the new points
        if points_to_eval_set:
            points_np = np.array(list(points_to_eval_set), dtype='f4')
            distances = sdf_callable(points_np)
            for i, p_tuple in enumerate(points_to_eval_set):
                point_cache[p_tuple] = distances[i]

        # 3. Determine which voxels to subdivide (now that all points are cached)
        surface_voxels_mask = np.zeros(len(voxels), dtype=bool)
        voxel_diagonals = np.linalg.norm(maxs - mins, axis=1)
        
        for i in range(len(voxels)):
            corner_dists = [point_cache[tuple(c)] for c in all_corners[i]]
            center_dist = point_cache[tuple(centers[i])]
            
            # Condition 1: Surface crosses boundary (sign change)
            if min(corner_dists) * max(corner_dists) <= 0:
                surface_voxels_mask[i] = True
            # Condition 2: Voxel is close to surface (handles contained objects)
            elif abs(center_dist) < voxel_diagonals[i] * 0.866: # sqrt(3)/2
                surface_voxels_mask[i] = True

        surface_voxels = voxels[surface_voxels_mask]
        
        if len(surface_voxels) == 0:
            if verbose: print("  - No surface intersections found, stopping subdivision.")
            return [], {}, 0
        
        if depth < max_depth - 1:
            mins = surface_voxels[:, 0]
            centers = (surface_voxels[:, 0] + surface_voxels[:, 1]) / 2.0
            maxs = surface_voxels[:, 1]
            
            mx, my, mz = mins[:,0], mins[:,1], mins[:,2]
            cx, cy, cz = centers[:,0], centers[:,1], centers[:,2]
            Mx, My, Mz = maxs[:,0], maxs[:,1], maxs[:,2]
            
            children_mins = [
                np.stack([mx, my, mz], 1), np.stack([cx, my, mz], 1),
                np.stack([mx, cy, mz], 1), np.stack([mx, my, cz], 1),
                np.stack([cx, cy, mz], 1), np.stack([cx, my, cz], 1),
                np.stack([mx, cy, cz], 1), np.stack([cx, cy, cz], 1),
            ]
            children_maxs = [
                np.stack([cx, cy, cz], 1), np.stack([Mx, cy, cz], 1),
                np.stack([cx, My, cz], 1), np.stack([cx, cy, Mz], 1),
                np.stack([Mx, My, cz], 1), np.stack([Mx, cy, Mz], 1),
                np.stack([cx, My, Mz], 1), np.stack([Mx, My, Mz], 1),
            ]
            
            all_children = [np.stack([m, M], axis=1) for m, M in zip(children_mins, children_maxs)]
            voxels = np.concatenate(all_children, axis=0)
        else:
            voxels = surface_voxels

    leaf_voxels = voxels
    if verbose:
        print(f"  - Octree evaluation complete. Found {len(leaf_voxels)} leaf voxels.")
        print(f"  - Total unique points evaluated: {len(point_cache)}")

    if len(leaf_voxels) == 0:
        return [], {}, 0
        
    grid_coords = np.array(list(point_cache.keys()))
    min_coord = np.min(grid_coords, axis=0)
    step = (leaf_voxels[0, 1, :] - leaf_voxels[0, 0, :])[0]
    
    indices = np.round((grid_coords - min_coord) / (step + 1e-9)).astype(int)
    max_indices = np.max(indices, axis=0)
    
    fill_value = np.linalg.norm(root_max - root_min)
    volume = np.full(max_indices + 1, fill_value, dtype='f4')
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = list(point_cache.values())
    
    return volume, min_coord, step

def _get_normal_callable(sdf_callable):
    """Creates a function to compute SDF normals using central differences."""
    def normal_callable(points):
        eps = 1e-5
        p = np.array(points, dtype='f4')
        dx = sdf_callable(p + np.array([eps, 0, 0])) - sdf_callable(p - np.array([eps, 0, 0]))
        dy = sdf_callable(p + np.array([0, eps, 0])) - sdf_callable(p - np.array([0, eps, 0]))
        dz = sdf_callable(p + np.array([0, 0, eps])) - sdf_callable(p - np.array([0, 0, eps]))
        
        normals = np.stack([dx, dy, dz], axis=-1)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals / np.where(norms == 0, 1, norms)
    return normal_callable

def _dual_contouring(sdf_callable, bounds, resolution, verbose):
    if verbose:
        print(f"  - Using Dual Contouring with resolution {resolution}.")

    # 1. Grid setup
    X = np.linspace(bounds[0][0], bounds[1][0], resolution[0])
    Y = np.linspace(bounds[0][1], bounds[1][1], resolution[1])
    Z = np.linspace(bounds[0][2], bounds[1][2], resolution[2])
    Nx, Ny, Nz = len(X), len(Y), len(Z)
    
    points_grid = _cartesian_product(X, Y, Z).astype('f4')
    if verbose:
        print(f"  - Grid dimensions: {Nx} x {Ny} x {Nz} = {len(points_grid)} points")
        print("  - Evaluating SDF on grid...")
    volume = sdf_callable(points_grid).reshape(Nx, Ny, Nz)
    
    normal_callable = _get_normal_callable(sdf_callable)

    # 2. Find feature points by solving QEFs
    if verbose: print("  - Finding feature points...")
    cell_to_vert_idx = {}
    verts = []
    
    cell_corners = np.array([ [i,j,k] for i in (0,1) for j in (0,1) for k in (0,1) ])
    edge_v_map = [[0,1], [2,3], [4,5], [6,7], [0,2], [1,3], [4,6], [5,7], [0,4], [1,5], [2,6], [3,7]]

    for i in range(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz - 1):
                # Get corner values for this cell
                corner_vals = volume[i:i+2, j:j+2, k:k+2].flatten()
                
                # If all corners have the same sign, cell is not on the surface
                if np.all(corner_vals > 0) or np.all(corner_vals < 0):
                    continue

                # Cell is on the surface, find edge intersections
                cross_points, cross_normals = [], []
                
                cell_origin = np.array([X[i], Y[j], Z[k]])
                cell_size = np.array([X[i+1]-X[i], Y[j+1]-Y[j], Z[k+1]-Z[k]])

                for v0_idx, v1_idx in edge_v_map:
                    p0_idx = cell_corners[v0_idx] + [i,j,k]
                    p1_idx = cell_corners[v1_idx] + [i,j,k]
                    val0, val1 = volume[tuple(p0_idx)], volume[tuple(p1_idx)]
                    
                    if val0 * val1 < 0:
                        p0 = np.array([X[p0_idx[0]], Y[p0_idx[1]], Z[p0_idx[2]]])
                        p1 = np.array([X[p1_idx[0]], Y[p1_idx[1]], Z[p1_idx[2]]])
                        
                        # Linear interpolation to find intersection
                        t = val0 / (val0 - val1)
                        intersect_p = p0 + t * (p1 - p0)
                        
                        cross_points.append(intersect_p)
                
                if not cross_points: continue
                
                # Get normals at intersection points
                cross_normals = normal_callable(np.array(cross_points))

                # Build and solve QEF
                A = np.sum([np.outer(n, n) for n in cross_normals], axis=0)
                b = np.sum([np.dot(n, p) * n for n, p in zip(cross_normals, cross_points)], axis=0)
                
                try:
                    # Use pseudo-inverse for robustness
                    vert = np.linalg.pinv(A) @ b
                    # Clamp vertex to be inside the cell
                    vert = np.clip(vert, cell_origin, cell_origin + cell_size)
                except np.linalg.LinAlgError:
                    vert = np.mean(cross_points, axis=0) # Fallback

                cell_to_vert_idx[(i,j,k)] = len(verts)
                verts.append(vert)

    # 3. Generate faces
    if verbose: print("  - Generating faces...")
    faces = []
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz - 1):
                # X-aligned edges
                if j > 0 and k > 0 and volume[i,j,k] * volume[i+1,j,k] < 0:
                    v1 = cell_to_vert_idx.get((i, j - 1, k - 1))
                    v2 = cell_to_vert_idx.get((i, j, k - 1))
                    v3 = cell_to_vert_idx.get((i, j, k))
                    v4 = cell_to_vert_idx.get((i, j - 1, k))
                    if all(v is not None for v in [v1,v2,v3,v4]):
                        if volume[i,j,k] < 0:
                            faces.extend([[v1,v2,v3], [v1,v3,v4]])
                        else:
                            faces.extend([[v1,v4,v3], [v1,v3,v2]])

                # Y-aligned edges
                if i > 0 and k > 0 and volume[i,j,k] * volume[i,j+1,k] < 0:
                    v1 = cell_to_vert_idx.get((i - 1, j, k - 1))
                    v2 = cell_to_vert_idx.get((i, j, k - 1))
                    v3 = cell_to_vert_idx.get((i, j, k))
                    v4 = cell_to_vert_idx.get((i - 1, j, k))
                    if all(v is not None for v in [v1,v2,v3,v4]):
                        if volume[i,j,k] < 0:
                            faces.extend([[v1,v4,v3], [v1,v3,v2]])
                        else:
                            faces.extend([[v1,v2,v3], [v1,v3,v4]])

                # Z-aligned edges
                if i > 0 and j > 0 and volume[i,j,k] * volume[i,j,k+1] < 0:
                    v1 = cell_to_vert_idx.get((i - 1, j - 1, k))
                    v2 = cell_to_vert_idx.get((i, j - 1, k))
                    v3 = cell_to_vert_idx.get((i, j, k))
                    v4 = cell_to_vert_idx.get((i - 1, j, k))
                    if all(v is not None for v in [v1,v2,v3,v4]):
                        if volume[i,j,k] < 0:
                            faces.extend([[v1,v2,v3], [v1,v3,v4]])
                        else:
                            faces.extend([[v1,v4,v3], [v1,v3,v2]])
                            
    return np.array(verts), np.array(faces)

def generate(sdf_obj, bounds=None, samples=2**22, algorithm='marching_cubes', adaptive=False, octree_depth=8, decimate_ratio=None, verbose=True):
    """
    Generates a mesh (vertices and faces) from an SDF object.
    
    Args:
        sdf_obj (SDFNode): The object to mesh.
        bounds (tuple, optional): The bounding box. Estimated if None.
        samples (int, optional): Number of samples for uniform grid.
        algorithm (str, optional): 'marching_cubes' or 'dual_contouring'.
        adaptive (bool, optional): Use adaptive octree meshing.
        octree_depth (int, optional): Depth for adaptive meshing.
        decimate_ratio (float, optional): Ratio of triangles to remove (0.0 - 1.0).
        verbose (bool, optional): Print status.

    Returns:
        tuple: (verts, faces) where verts is (N, 3) float32 and faces is (M, 3) int.
    """
    if algorithm not in ['marching_cubes', 'dual_contouring']:
        print(f"WARNING: Algorithm '{algorithm}' is not supported. Falling back to 'marching_cubes'.", file=sys.stderr)
        algorithm = 'marching_cubes'

    if bounds is None:
        if verbose:
            print("INFO: No bounds provided, estimating automatically.", file=sys.stderr)
        bounds = sdf_obj.estimate_bounds(verbose=verbose)

    if verbose:
        print(f"INFO: Generating mesh...")
        print(f"  - Bounds: {bounds}")

    try:
        sdf_callable = sdf_obj.to_callable()
    except (TypeError, NotImplementedError, ImportError) as e:
        print(f"ERROR: Could not generate mesh. {e}")
        raise

    verts, faces = [], []
    
    if adaptive:
        if algorithm == 'dual_contouring':
            raise ValueError("Dual Contouring algorithm does not currently support adaptive meshing. Please set `adaptive=False`.")

        if samples != 2**22:
             print(f"WARNING: `samples` parameter is ignored when `adaptive=True`. Using `octree_depth={octree_depth}` instead.", file=sys.stderr)

        volume, origin, step_size = _adaptive_meshing(sdf_callable, bounds, octree_depth, verbose)
        if len(volume) > 0:
            try:
                verts, faces, _, _ = measure.marching_cubes(volume, level=0, spacing=(step_size, step_size, step_size))
                verts += origin
            except (ValueError, RuntimeError):
                verts = []
    elif algorithm == 'dual_contouring':
        res = int(round(samples**(1/3)))
        if res < 4:
            print(f"WARNING: samples={samples} results in a very low grid resolution ({res}). Mesh quality may be poor. Increasing resolution.", file=sys.stderr)
            res = 4
        verts, faces = _dual_contouring(sdf_callable, bounds, (res, res, res), verbose)
    else: # Uniform marching cubes
        if octree_depth != 8:
             print("WARNING: `octree_depth` parameter is ignored when `adaptive=False`. Using `samples` instead.", file=sys.stderr)
        
        volume_size = (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1]) * (bounds[1][2] - bounds[0][2])
        step = (volume_size / samples) ** (1 / 3)

        if verbose:
            print(f"  - Target samples: {samples}")
            print(f"  - Voxel step size: {step:.4f}")

        X = np.arange(bounds[0][0], bounds[1][0], step)
        Y = np.arange(bounds[0][1], bounds[1][1], step)
        Z = np.arange(bounds[0][2], bounds[1][2], step)

        if verbose:
            count = len(X)*len(Y)*len(Z)
            print(f"  - Grid dimensions: {len(X)} x {len(Y)} x {len(Z)} = {count} points")

        points_grid = _cartesian_product(X, Y, Z).astype('f4')

        if verbose:
            print("  - Evaluating SDF on grid...")

        distances = sdf_callable(points_grid)
        distances = np.array(distances, dtype='f4').reshape(len(X), len(Y), len(Z))
        
        try:
            verts, faces, _, _ = measure.marching_cubes(distances, level=0, spacing=(step, step, step))
            verts += np.array(bounds[0])
        except ValueError:
            verts = []

    if len(verts) == 0 or len(faces) == 0:
        print("ERROR: Mesh generation failed. The surface may not intersect the specified bounds or the algorithm returned no geometry.", file=sys.stderr)
        return np.array([]), np.array([])

    if decimate_ratio is not None:
        if not (0 < decimate_ratio < 1):
            print("WARNING: `decimate_ratio` must be between 0 and 1. Skipping simplification.", file=sys.stderr)
        else:
            try:
                import trimesh
                if verbose:
                    print(f"  - Simplifying mesh (target reduction: {decimate_ratio * 100:.1f}%)...")
                
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                
                original_face_count = len(mesh.faces)
                target_face_count = int(original_face_count * (1.0 - decimate_ratio))

                simplified_mesh = mesh.simplify_quadric_decimation(target_face_count)

                verts = simplified_mesh.vertices
                faces = simplified_mesh.faces
                if verbose:
                    print(f"  - Simplified from {original_face_count} to {len(faces)} faces.")

            except ImportError:
                print("WARNING: Mesh simplification requires the 'trimesh' library.", file=sys.stderr)
                print("         Please install it via: pip install trimesh", file=sys.stderr)
            except Exception as e:
                print(f"ERROR: Mesh simplification failed: {e}", file=sys.stderr)

    return verts, faces

def save(sdf_obj, path, bounds, samples, verbose, algorithm, adaptive, vertex_colors, decimate_ratio=None, octree_depth=8):
    start_time = time.time()
    
    verts, faces = generate(sdf_obj, bounds, samples, algorithm, adaptive, octree_depth, decimate_ratio, verbose)
    
    if len(verts) == 0 or len(faces) == 0:
        return

    if verbose:
        print(f"INFO: Saving to '{path}'...")

    path_lower = path.lower()
    if path_lower.endswith('.stl'):
        _write_binary_stl(path, verts[faces])
    elif path_lower.endswith('.obj'):
        _write_obj(path, verts, faces)
    elif path_lower.endswith('.glb') or path_lower.endswith('.gltf'):
        if vertex_colors:
            print("WARNING: vertex_colors=True is not yet implemented for GLB export.", file=sys.stderr)
        _write_glb(path, verts, faces, vertex_colors)
    else:
        print(f"ERROR: Unsupported file format '{path}'. Only .stl, .obj, .glb, and .gltf are currently supported.", file=sys.stderr)
        return

    elapsed = time.time() - start_time
    if verbose:
        print(f"SUCCESS: Mesh with {len(faces)} triangles saved to '{path}' in {elapsed:.2f}s.")