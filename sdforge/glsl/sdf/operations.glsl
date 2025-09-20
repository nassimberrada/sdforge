// --- Standard Boolean Operations ---

vec4 opU(vec4 a, vec4 b) {
    return (a.x < b.x) ? a : b;
}

vec4 opI(vec4 a, vec4 b) {
    return (a.x > b.x) ? a : b;
}

vec4 opS(vec4 a, vec4 b) {
    return opI(a, vec4(-b.x, b.y, b.z, b.w));
}

vec4 opX(vec4 a, vec4 b) {
    return opI(opU(a,b), vec4(-opI(a,b).x, opI(a,b).y, 0, 0));
}


// --- Smooth Boolean Operations ---

vec4 sUnion(vec4 a, vec4 b, float k ) 
{
    float h = clamp( 0.5 + 0.5*(b.x-a.x)/k, 0.0, 1.0 );
    float dist = mix( b.x, a.x, h ) - k*h*(1.0-h);
    return (a.x < b.x) ? vec4(dist, a.y, a.z, a.w) : vec4(dist, b.y, b.z, b.w);
}

vec4 sIntersect(vec4 a, vec4 b, float k ) 
{
    float h = clamp( 0.5 - 0.5*(b.x-a.x)/k, 0.0, 1.0 );
    float dist = mix( b.x, a.x, h ) + k*h*(1.0-h);
    return (a.x > b.x) ? vec4(dist, a.y, a.z, a.w) : vec4(dist, b.y, b.z, b.w);
}

vec4 sDifference(vec4 a, vec4 b, float k ) 
{
    float h = clamp( 0.5 - 0.5*(b.x+a.x)/k, 0.0, 1.0 );
    float dist = mix( a.x, -b.x, h ) + k*h*(1.0-h);
    return vec4(dist, a.y, a.z, a.w);
}

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