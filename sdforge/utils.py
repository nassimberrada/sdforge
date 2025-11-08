def _glsl_format(val):
    """Formats a Python value for injection into a GLSL string."""
    # Check for Param first as it has a custom handler
    if hasattr(val, 'to_glsl'):
        return val.to_glsl()
    if isinstance(val, str):
        return val  # Assume it's a raw GLSL expression
    return f"{float(val)}"
