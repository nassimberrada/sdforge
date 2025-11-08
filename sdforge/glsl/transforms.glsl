// <file_path:/home/nb/Projects/sdforge/sdforge/glsl/transforms.glsl>
// --- Transformations ---
// For SDFs, we apply the INVERSE transformation to the point 'p'.

vec3 opTranslate(vec3 p, vec3 offset) {
    return p - offset;
}

// For non-uniform scaling, the distance must be corrected.
// The GLSL for that is handled in the Python Scale class.
vec3 opScale(vec3 p, vec3 factor) {
    return p / factor;
}

vec3 opRotateX(vec3 p, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  // Apply inverse rotation
  return vec3(p.x, p.y * c + p.z * s, -p.y * s + p.z * c);
}

vec3 opRotateY(vec3 p, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  // Apply inverse rotation
  return vec3(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
}

vec3 opRotateZ(vec3 p, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  // Apply inverse rotation
  return vec3(p.x * c + p.y * s, -p.x * s + p.y * c, p.z);
}

vec3 opTwist(vec3 p, float k)
{
    // Inverse twist
    float c = cos(-k*p.y);
    float s = sin(-k*p.y);
    mat2 m = mat2(c, -s, s, c);
    p.xz = m * p.xz;
    return p;
}

vec3 opShearXY(vec3 p, vec2 shear) {
    // Inverse shear
    return vec3(p.x - shear.x * p.z, p.y - shear.y * p.z, p.z);
}

vec3 opShearXZ(vec3 p, vec2 shear) {
    // Inverse shear
    return vec3(p.x - shear.x * p.y, p.y, p.z - shear.y * p.y);
}

vec3 opShearYZ(vec3 p, vec2 shear) {
    // Inverse shear
    return vec3(p.x, p.y - shear.x * p.x, p.z - shear.y * p.x);
}

vec3 opBendX(vec3 p, float k) {
    // Inverse bend
    float c = cos(k * p.x);
    float s = sin(k * p.x);
    return vec3(p.x, c * p.y + s * p.z, -s * p.y + c * p.z);
}

vec3 opBendY(vec3 p, float k) {
    // Inverse bend
    float c = cos(k * p.y);
    float s = sin(k * p.y);
    return vec3(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

vec3 opBendZ(vec3 p, float k) {
    // Inverse bend
    float c = cos(k * p.z);
    float s = sin(k * p.z);
    return vec3(c * p.x + s * p.y, -s * p.x + c * p.y, p.z);
}

// --- Domain Repetition ---

vec3 opRepeat(vec3 p, vec3 c)
{
    return mod(p + 0.5 * c, c) - 0.5 * c;
}

vec3 opLimitedRepeat(vec3 p, vec3 s, vec3 l) {
    return p - s * clamp(round(p / s), -l, l);
}

vec3 opPolarRepeat(vec3 p, float repetitions) {
    float angle = 2.0 * 3.14159265 / repetitions;
    float a = atan(p.x, p.z);
    float r = length(p.xz);
    float newA = mod(a, angle) - 0.5 * angle;
    return vec3(r * sin(newA), p.y, r * cos(newA));
}

vec3 opMirror(vec3 p, vec3 a) {
    if (a.x > 0.5) p.x = abs(p.x);
    if (a.y > 0.5) p.y = abs(p.y);
    if (a.z > 0.5) p.z = abs(p.z);
    return p;
}