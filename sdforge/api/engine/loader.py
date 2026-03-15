from pathlib import Path
from functools import lru_cache

# A dictionary to hold all loaded GLSL file contents, mapping stem -> full_code
GLSL_SOURCES = {}

# Defines the order in which GLSL files should be concatenated to satisfy dependencies.
# Files not in this list will be appended alphabetically after these.
GLSL_ORDER = [
    'debug',      # Standalone
    'combinations', # Boolean math, min/max
    'shaping',    # Rounding, displacement
    'transforms', # Depends on noise
    'geometry',   # Base shapes
    'camera',     # Render logic
    'light',      # Render logic
    'raymarching' # Render logic
]

# Define implicit dependencies: key implies values must also be included.
# This handles cases where a file grows a dependency
# that isn't explicitly listed in every Python node class.
IMPLICIT_DEPENDENCIES = {
    'transforms': {'shaping'},
}

def load_all_glsl():
    """Finds and loads all .glsl files into a dictionary."""
    if GLSL_SOURCES:
        return

    glsl_dir = Path(__file__).parent.parent / 'glsl'
    
    # If not found, check one level higher (the main `sdforge` directory)
    if not glsl_dir.exists():
        glsl_dir = Path(__file__).parent.parent.parent / 'glsl'
        
        # If still not found, raise an error instead of failing silently
        if not glsl_dir.exists():
            raise FileNotFoundError(
                f"Could not find the 'glsl' library directory. Looked in:\n"
                f"- {Path(__file__).parent.parent / 'glsl'}\n"
                f"- {glsl_dir}"
            )

    for glsl_file in glsl_dir.glob('*.glsl'):
        with open(glsl_file, 'r') as f:
            # Key is the filename without extension (e.g., 'geometry')
            GLSL_SOURCES[glsl_file.stem] = f.read()

@lru_cache(maxsize=None)
def get_glsl_definitions(required_files: frozenset) -> str:
    """
    Given a set of required file stems, returns a single string
    containing all necessary GLSL code blocks in the correct dependency order.
    """
    if not GLSL_SOURCES:
        load_all_glsl()
    
    # 1. Expand dependencies
    expanded_files = set(required_files)
    # Simple one-pass expansion handles current depth-1 dependencies.
    for req in required_files:
        if req in IMPLICIT_DEPENDENCIES:
            expanded_files.update(IMPLICIT_DEPENDENCIES[req])

    # 2. Sort required files based on GLSL_ORDER
    def sort_key(name):
        try:
            return GLSL_ORDER.index(name)
        except ValueError:
            return len(GLSL_ORDER) + 1 # Push unknown files to the end

    sorted_files = sorted(list(expanded_files), key=sort_key)

    code_blocks = [
        GLSL_SOURCES[stem] for stem in sorted_files if stem in GLSL_SOURCES
    ]
    return "\n\n".join(code_blocks)