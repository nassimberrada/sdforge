// --- Shaping Operations ---
vec4 opRound(vec4 res, float r) {
    res.x -= r;
    return res;
}

vec4 opBevel(vec4 res, float thickness) {
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