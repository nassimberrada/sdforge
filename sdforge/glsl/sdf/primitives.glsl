// sdforge/glsl/sdf/primitives.glsl

// --- 3D Primitives ---

float sdSphere(in vec3 p, in float r) {
    return length(p) - r;
}

float sdBox(in vec3 p, in vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdRoundedBox(in vec3 p, in vec3 b, in float r) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) - r;
}

float sdTorus(in vec3 p, in vec2 t) { // t.x=major radius, t.y=minor radius
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdCapsule(in vec3 p, in vec3 a, in vec3 b, in float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdCone(in vec3 p, in vec2 c) { // c.x=height, c.y=radius
    vec2 q = vec2(length(p.xz), p.y);
    vec2 a = c - q;
    vec2 b = q - c * vec2(1, -1);
    float k = sign(c.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (q.x * c.y - q.y * c.x), k * (q.y - c.y));
    return sqrt(d) * sign(s);
}

float sdPlane(in vec3 p, in vec4 n) { // n.xyz is normal, n.w is offset
    return dot(p, n.xyz) + n.w;
}

float sdHexPrism(in vec3 p, in vec2 h) { // h.x=radius, h.y=height
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735026);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    vec2 d = vec2(
         length(p.xy - vec2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x),
         p.z - h.y);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdOctahedron(in vec3 p, in float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

float sdEllipsoid(in vec3 p, in vec3 r) { // r is radii on each axis
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float sdCylinder(vec3 p, vec2 h) { // h.x=radius, h.y=half-height
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// --- 2D Primitives (for extrusion or direct use if p.z=0) ---

float sdCircle(in vec2 p, in float r) {
    return length(p) - r;
}

float sdRectangle(in vec2 p, in vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}