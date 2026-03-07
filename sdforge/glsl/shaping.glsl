// --- Shaping Operations ---
vec4 opRound(vec4 res, float r) {
    res.x -= r;
    return res;
}

vec4 opShell(vec4 res, float thickness) {
    res.x = abs(res.x) - thickness;
    return res;
}

// --- Displacement ---
vec4 opDisplace(vec4 res, float displacement) {
    res.x += displacement;
    return res;
}

// --- Extrusion ---
vec4 opExtrude(vec4 res, vec3 p, float h) {
    vec2 w = vec2(res.x, abs(p.z) - h);
    res.x = min(max(w.x, w.y), 0.0) + length(max(w, 0.0));
    return res;
}

// --- Revolution ---
// --- Shaping Operations ---
vec3 opRevolve(vec3 p, vec3 axis, float side) {
    vec3 v = normalize(axis);
    float h = dot(p, v);
    vec3 p_perp = p - h * v;
    float r = length(p_perp);
    
    vec2 v_proj = vec2(v.x, v.y);
    float proj_len = length(v_proj);
    
    if (proj_len > 1e-5) {
        // Map perfectly back to the XY plane
        vec2 v2d = v_proj / proj_len;
        vec2 u2d = vec2(-v2d.y, v2d.x);
        vec2 mapped = h * v2d + side * r * u2d;
        return vec3(mapped, 0.0);
    } else {
        // Fallback for Z-axis
        return vec3(side * r, h, 0.0);
    }
}