import numpy as np
import time
import struct
import sys
from skimage import measure

def _cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

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


def save(sdf_obj, path, bounds, samples, verbose, algorithm, adaptive, vertex_colors):
    """
    Generates a mesh from an SDF object using the Marching Cubes algorithm and saves it to a file.
    """
    if algorithm != 'marching_cubes':
        print(f"WARNING: Algorithm '{algorithm}' is not supported. Falling back to 'marching_cubes'.", file=sys.stderr)
    if adaptive:
        print("WARNING: Adaptive meshing is not yet implemented. Using uniform grid.", file=sys.stderr)

    start_time = time.time()
    if verbose:
        print(f"INFO: Generating mesh for '{path}'...")

    try:
        sdf_callable = sdf_obj.to_callable()
    except (TypeError, NotImplementedError, ImportError) as e:
        print(f"ERROR: Could not generate mesh. {e}")
        raise

    volume = (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1]) * (bounds[1][2] - bounds[0][2])
    step = (volume / samples) ** (1 / 3)

    if verbose:
        print(f"  - Bounds: {bounds}")
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
    except ValueError:
        print("ERROR: Marching cubes failed. The surface may not intersect the specified bounds or the SDF evaluation returned invalid values.", file=sys.stderr)
        return
        
    verts += np.array(bounds[0])

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