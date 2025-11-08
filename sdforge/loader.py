from pathlib import Path
from functools import lru_cache

# A dictionary to hold all loaded GLSL file contents, mapping stem -> full_code
GLSL_SOURCES = {}

def load_all_glsl():
    """Finds and loads all .glsl files into a dictionary."""
    if GLSL_SOURCES:  # Only load once
        return

    glsl_dir = Path(__file__).parent / 'glsl'
    if not glsl_dir.exists(): return

    for glsl_file in glsl_dir.glob('*.glsl'):
        with open(glsl_file, 'r') as f:
            # Key is the filename without extension (e.g., 'noise')
            GLSL_SOURCES[glsl_file.stem] = f.read()

@lru_cache(maxsize=None)
def get_glsl_definitions(required_files: frozenset) -> str:
    """
    Given a set of required file stems, returns a single string
    containing all necessary GLSL code blocks.
    """
    if not GLSL_SOURCES:
        load_all_glsl()
    
    code_blocks = [
        GLSL_SOURCES[stem] for stem in required_files if stem in GLSL_SOURCES
    ]
    return "\n\n".join(code_blocks)