import os
import shutil
import subprocess
import tempfile
import inspect
import re

import pytest
import sdforge

# ---- Configuration ----
GLSL_VALIDATOR = shutil.which("glslangValidator")
# If you want to skip GLSL checks locally/CI set env var SKIP_GLSL=1
SKIP_GLSL = os.environ.get("SKIP_GLSL", "") == "1"

if GLSL_VALIDATOR is None and not SKIP_GLSL:
    pytest.skip("glslangValidator not found on PATH. Set SKIP_GLSL=1 to skip GLSL validation.", allow_module_level=True)


# ---- Helpers to find & create primitives ----

def get_simple_factories():
    """
    Find callable factories in sdforge that don't require positional arguments.
    Returns list of tuples (name, callable)
    """
    factories = []
    for name, obj in vars(sdforge).items():
        if not callable(obj):
            continue
        # skip internal / capitalized constructors (we want factory functions)
        if name[0].isupper():
            continue
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue
        # skip if function explicitly marked private
        if name.startswith("_"):
            continue
        # include only if no required (positional-only / positional-or-keyword) parameters
        params = [p for p in sig.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if not params:
            factories.append((name, obj))
    return factories


# Some factories require arguments (no defaults). Provide sensible overrides here.
# Add other factories here as needed.
OVERRIDES = {
    "capsule": lambda: sdforge.capsule((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), radius=0.15),
    "ellipsoid": lambda: sdforge.ellipsoid((1.0, 0.6, 0.8)),
    "box_frame": lambda: sdforge.box_frame((1.0, 1.0, 1.0), edge_radius=0.08),
    "capped_torus": lambda: sdforge.capped_torus((0.0, 1.0), 1.0, 0.25),
    "capped_cylinder": lambda: sdforge.capped_cylinder((0.0, 0.0, -0.5), (0.0, 0.0, 0.5), 0.1),
    "capped_cone": lambda: sdforge.capped_cone(1.0, 0.5, 0.2),
    # If your project exposes other factories that need args, add them here.
}


def collect_factories():
    """Return a dict name->callable factory (call returns SDFObject)."""
    factories = dict(get_simple_factories())
    # include overrides (will overwrite if same name exists)
    factories.update(OVERRIDES)
    # Sanity: filter out anything not callable (defensive)
    return {n: f for n, f in factories.items() if callable(f)}


# ---- GLSL preamble (stubs & helper functions) ----
# We include minimal implementations / stubs for common sd functions and op functions
# so glslangValidator can parse and compile the shader. They don't need to be
# physically correct, just syntactically correct and present.
GLSL_PREAMBLE = r"""
#version 330 core
// Basic stub implementations of commonly referenced SDF functions.
// These implementations are intentionally simple (correctness not required),
// only their presence/signatures matter for syntax checking.

// Primitive SDFs (signatures used by api.py)
float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, vec3(0.0))) + min(max(max(q.x, q.y), q.z), 0.0);
}
float sdRoundedBox(vec3 p, vec3 b, float r) { return sdBox(p, b) - r; }
float sdTorus(vec3 p, vec2 t) { vec2 q = vec2(length(p.xz) - t.x, p.y); return length(q) - t.y; }
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) { return 0.0; }
float sdCylinder(vec3 p, vec2 h) { return max(length(p.xz) - h.x, abs(p.y) - h.y); }
float sdCone(vec3 p, vec2 c) { return 0.0; }
float sdPlane(vec3 p, vec4 eq) { return dot(p, eq.xyz) + eq.w; }
float sdHexPrism(vec3 p, vec2 h) { return 0.0; }
float sdOctahedron(vec3 p, float s) { return (abs(p.x) + abs(p.y) + abs(p.z) - s) * 0.57735027; }
float sdEllipsoid(vec3 p, vec3 r) { vec3 q = p / r; return length(q) - 1.0; }
float sdBoxFrame(vec3 p, vec3 b, float e) { return 0.0; }
float sdCappedTorus(vec3 p, vec2 sc, float ra, float rb) { return 0.0; }
float sdLink(vec3 p, float le, float r1, float r2) { return 0.0; }
float sdCappedCylinder(vec3 p, vec3 a, vec3 b, float r) { return 0.0; }
float sdRoundedCylinder(vec3 p, float ra, float rb, float h) { return 0.0; }
float sdCappedCone(vec3 p, float h, float r1, float r2) { return 0.0; }
float sdRoundCone(vec3 p, float r1, float r2, float h) { return 0.0; }
float sdPyramid(vec3 p, float h) { return 0.0; }

// Operation helpers (vec4-based SDF result: x=distance, y=material id, z,w reserved)
vec4 opU(vec4 a, vec4 b) { return a; }
vec4 opI(vec4 a, vec4 b) { return a; }
vec4 opS(vec4 a, vec4 b) { return a; }
vec4 opX(vec4 a, vec4 b) { return a; }

vec4 sUnion(vec4 a, vec4 b, float k) { return a; }
vec4 sIntersect(vec4 a, vec4 b, float k) { return a; }
vec4 sDifference(vec4 a, vec4 b, float k) { return a; }

vec4 opRound(vec4 a, float r) { return a; }
vec4 opBevel(vec4 a, float t) { return a; }
vec4 opDisplace(vec4 a, float d) { return a; }
vec4 opExtrude(vec4 a, vec3 p, float h) { return a; }

// Transforms / helpers
vec3 opTranslate(vec3 p, vec3 o) { return p - o; }
vec3 opScale(vec3 p, vec3 f) { return p / f; }
vec3 opRotateX(vec3 p, float a) { return p; }
vec3 opRotateY(vec3 p, float a) { return p; }
vec3 opRotateZ(vec3 p, float a) { return p; }
vec3 opTwist(vec3 p, float k) { return p; }
vec3 opShearXY(vec3 p, vec2 s) { return p; }
vec3 opShearXZ(vec3 p, vec2 s) { return p; }
vec3 opShearYZ(vec3 p, vec2 s) { return p; }
vec3 opBendX(vec3 p, float k) { return p; }
vec3 opBendY(vec3 p, float k) { return p; }
vec3 opBendZ(vec3 p, float k) { return p; }
vec3 opElongate(vec3 p, vec3 h) { return p; }
vec3 opRepeat(vec3 p, vec3 s) { return p; }
vec3 opLimitedRepeat(vec3 p, vec3 s, vec3 l) { return p; }
vec3 opPolarRepeat(vec3 p, float n) { return p; }
vec3 opMirror(vec3 p, vec3 a) { return p; }


// Fallbacks for other custom op names (safe no-op)
vec4 opSmoothUnion(vec4 a, vec4 b, float k) { return a; }
vec4 opSmoothIntersection(vec4 a, vec4 b, float k) { return a; }
vec4 opSmoothDifference(vec4 a, vec4 b, float k) { return a; }
"""

# ---- GLSL validator wrapper ----

def validate_glsl(glsl_code: str):
    """
    Wrap glsl_code into a tiny fragment shader and run glslangValidator.
    Raises an AssertionError if compilation fails.
    """
    # Ensure the code ends up as a single expression returning a vec4 called in eval(p)
    shader = GLSL_PREAMBLE + "\n" + f"""
vec4 eval(vec3 p) {{
    return {glsl_code};
}}

void main() {{
    vec3 p = vec3(0.0);
    vec4 v = eval(p);
}}
"""
    with tempfile.NamedTemporaryFile(suffix=".frag", mode="w", delete=False) as f:
        f.write(shader)
        path = f.name

    try:
        result = subprocess.run(
            [GLSL_VALIDATOR, "-S", "frag", path],
            capture_output=True, text=True
        )
    finally:
        # keep file for debugging if compilation failed (user can inspect).
        pass

    if result.returncode != 0:
        # include the shader snippet in the assertion to make debugging easier
        # trim very long code for readability
        snippet = glsl_code if len(glsl_code) < 4000 else (glsl_code[:4000] + "\n... (truncated)")
        raise AssertionError(f"glslangValidator failed (exit {result.returncode}).\n"
                             f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n\nGLSL snippet:\n{snippet}")


# ---- Variation generation ----

from math import pi
from itertools import islice

def make_variations(factory_callable):
    """
    Given a factory function that returns an SDFObject, produce a list of
    variations exercising transforms, materials, boolean ops, shaping, repeats, etc.
    """
    base = factory_callable()
    variations = []
    # plain object
    variations.append(base)

    # transforms
    try:
        variations.append(base.translate((0.1, 0.2, 0.3)))
    except Exception:
        pass
    try:
        variations.append(base.translate((0.0, 0.0, 0.5)).color(0.2, 0.5, 0.9))
    except Exception:
        pass
    try:
        variations.append(base.scale(1.2))
    except Exception:
        pass
    try:
        variations.append(base.rotate(sdforge.Z, 0.4))
    except Exception:
        pass
    try:
        variations.append(base.orient(sdforge.X))
    except Exception:
        pass
    try:
        variations.append(base.twist(2.0))
    except Exception:
        pass
    try:
        variations.append(base.shear_xy((0.1, 0.0)))
    except Exception:
        pass
    try:
        variations.append(base.bend_y(0.5))
    except Exception:
        pass

    # repeats / mirror
    try:
        variations.append(base.repeat((1.0, 0.0, 0.0)))
    except Exception:
        pass
    try:
        variations.append(base.limited_repeat((0.8, 0.0, 0.0), (3, 0, 0)))
    except Exception:
        pass
    try:
        variations.append(base.polar_repeat(8))
    except Exception:
        pass
    try:
        variations.append(base.mirror(sdforge.X))
    except Exception:
        pass

    # shaping ops
    try:
        variations.append(base.round(0.05))
    except Exception:
        pass
    try:
        variations.append(base.bevel(0.05))
    except Exception:
        pass
    try:
        # displacement uses a GLSL expression string internally, ensure to_glsl exists
        variations.append(base.displace("sin(p.x*10.0)*0.02"))
    except Exception:
        pass
    try:
        variations.append(base.extrude(0.2))
    except Exception:
        pass
    try:
        variations.append(base.elongate((0.5, 0.0, 0.5)))
    except Exception:
        pass

    # smooth ops with another small primitive
    other = sdforge.sphere(0.3)
    try:
        variations.append(base.smooth_union(other, 0.2))
    except Exception:
        pass
    try:
        variations.append(base.smooth_intersection(other, 0.2))
    except Exception:
        pass
    try:
        variations.append(base.smooth_difference(other, 0.2))
    except Exception:
        pass

    # boolean ops
    try:
        variations.append(base | other)
    except Exception:
        pass
    try:
        variations.append(base & other)
    except Exception:
        pass
    try:
        variations.append(base - other)
    except Exception:
        pass
    try:
        variations.append(base.xor(other))
    except Exception:
        pass

    # combined / material
    try:
        variations.append((base | sdforge.box(0.5)).color(1.0, 0.4, 0.1).translate((0.0, 0.0, 0.2)))
    except Exception:
        pass

    # keep variations unique-ish and limit count to avoid huge runtime
    unique = []
    seen = set()
    for v in variations:
        try:
            key = v.to_glsl()  # use generated GLSL as a uniqueness key
        except Exception:
            # If to_glsl raises here, we still want the test to fail later with an informative message,
            # so include the object directly (it will be attempted again in tests)
            key = repr(v)
        if key not in seen:
            seen.add(key)
            unique.append(v)
        if len(unique) >= 30:
            break
    return unique


# ---- The test: iterate factories -> variations -> validate GLSL ----

FACTORIES = collect_factories()
# Very small sanity: ensure we have at least a handful
assert FACTORIES, "No factories discovered from sdforge. Check sdforge.__init__.py exports."

@pytest.mark.parametrize("name,factory", sorted(FACTORIES.items()))
def test_factory_glsl_compiles(name, factory):
    """For a given factory, make many variations and assert the generated GLSL compiles."""
    variations = list(make_variations(factory))
    assert variations, f"No variations created for factory '{name}'."

    for i, obj in enumerate(variations):
        # Call to_glsl and then validate via glslangValidator
        try:
            glsl_code = obj.to_glsl()
        except Exception as exc:
            pytest.fail(f"to_glsl() raised for factory '{name}' variation #{i}: {exc}")

        # Quick sanity: ensure it's a vec4 expression (most of our API returns vec4 expressions)
        if not re.search(r"\bvec4\b", glsl_code) and not glsl_code.strip().startswith("vec4"):
            # It's not strictly required to contain 'vec4' text (some implementations may omit), so we only warn.
            pass

        # Run the external GLSL validator (if available)
        if GLSL_VALIDATOR and not SKIP_GLSL:
            try:
                validate_glsl(glsl_code)
            except AssertionError as err:
                # Attach factory+variation info and re-raise so pytest reports informative message
                pytest.fail(f"GLSL compile error for factory '{name}' variation #{i}:\n{err}")