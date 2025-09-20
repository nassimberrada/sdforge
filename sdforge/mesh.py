import numpy as np
import time
import struct
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

def save(sdf_obj, path, bounds=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), samples=2**22, verbose=True):
    """
    Generates a mesh from an SDF object using the Marching Cubes algorithm and saves it to a file.
    """
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

    # Create all grid points (float32 for downstream libs)
    points_grid = _cartesian_product(X, Y, Z).astype('f4')

    if verbose:
        print("  - Evaluating SDF on grid...")

    distances = sdf_callable(points_grid)
    distances = np.array(distances, dtype='f4').reshape(len(X), len(Y), len(Z))

    try:
        verts, faces, _, _ = measure.marching_cubes(distances, level=0, spacing=(step, step, step))
    except ValueError:
        print("ERROR: Marching cubes failed. The surface may not intersect the specified bounds or the SDF evaluation returned invalid values.")
        print("  Suggestions:")
        print("    - Increase 'samples' to produce a finer voxel grid (e.g. samples=2**23 or 2**24).")
        print("    - Expand 'bounds' so the object is definitely inside the volume.")
        print("    - Verify your SDF callable returns finite numeric distances for all points.")
        return
        
    verts += np.array(bounds[0]) # Offset vertices to correct world position

    if path.lower().endswith('.stl'):
        _write_binary_stl(path, verts[faces])
    else:
        print(f"ERROR: Unsupported file format '{path}'. Only .stl is currently supported.")
        return

    elapsed = time.time() - start_time
    if verbose:
        print(f"SUCCESS: Mesh with {len(faces)} triangles saved to '{path}' in {elapsed:.2f}s.")