import re
from pathlib import Path
from functools import lru_cache

# A dictionary to hold all parsed GLSL functions, mapping name -> full_code
GLSL_FUNCTIONS = {}

def parse_glsl_file(content: str):
    """Parses a GLSL file into a dictionary of functions."""
    # Regex to find function signatures (including return type) and their bodies
    func_pattern = re.compile(
        r"([\w\s]+\s+(\w+)\s*\([^)]*\))\s*\{([\s\S]*?)\}", re.MULTILINE
    )
    for match in func_pattern.finditer(content):
        signature = match.group(1).strip()
        name = match.group(2)
        body = match.group(3).strip()
        full_code = f"{signature} {{\n    {body}\n}}"
        GLSL_FUNCTIONS[name] = full_code

def load_all_glsl():
    """Finds and parses all .glsl files in the glsl/ directory."""
    glsl_dir = Path(__file__).parent / 'glsl'
    if not glsl_dir.exists(): return

    for glsl_file in glsl_dir.glob('*.glsl'):
        with open(glsl_file, 'r') as f:
            parse_glsl_file(f.read())

@lru_cache(maxsize=None)
def get_glsl_definitions(required_names: frozenset) -> str:
    """
    Given a set of required function names, returns a single string
    containing all necessary GLSL function definitions.
    """
    if not GLSL_FUNCTIONS:
        load_all_glsl()

    # In the future, this is where transitive dependency resolution would happen.
    # For now, it's a direct lookup.
    
    code_blocks = [
        GLSL_FUNCTIONS[name] for name in required_names if name in GLSL_FUNCTIONS
    ]
    return "\n\n".join(code_blocks)