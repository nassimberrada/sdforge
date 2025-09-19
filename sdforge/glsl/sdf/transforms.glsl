// --- Transformations ---

vec3 opTranslate(vec3 p, vec3 offset) {
    return p - offset;
}

vec3 opScale(vec3 p, vec3 factor) {
    return p / factor;
}

vec3 opScaleUniform(vec3 p, float s) {
    return p / s;
}

vec3 opOrient(vec3 p, int axis) {
    if (axis == 0) return p.zyx; // X
    if (axis == 1) return p.xzy; // Y
    return p; // Z (default, no-op)
}

vec3 opTwist(vec3 p, float k)
{
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    mat2  m = mat2(c,-s,s,c);
    p.xz = m*p.xz;
    return p;
}

vec3 opRotateX(vec3 p, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return vec3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

vec3 opRotateY(vec3 p, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

vec3 opRotateZ(vec3 p, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return vec3(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
}

vec3 opShearXY(vec3 p, vec2 shear) {
    // shear.x = factor in X wrt Z, shear.y = factor in Y wrt Z
    return vec3(p.x + shear.x * p.z, p.y + shear.y * p.z, p.z);
}

vec3 opShearXZ(vec3 p, vec2 shear) {
    // shear.x = factor in X wrt Y, shear.y = factor in Z wrt Y
    return vec3(p.x + shear.x * p.y, p.y, p.z + shear.y * p.y);
}

vec3 opShearYZ(vec3 p, vec2 shear) {
    // shear.x = factor in Y wrt X, shear.y = factor in Z wrt X
    return vec3(p.x, p.y + shear.x * p.x, p.z + shear.y * p.x);
}

vec3 opBendX(vec3 p, float k) {
    float c = cos(k * p.x);
    float s = sin(k * p.x);
    return vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

vec3 opBendY(vec3 p, float k) {
    float c = cos(k * p.y);
    float s = sin(k * p.y);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

vec3 opBendZ(vec3 p, float k) {
    float c = cos(k * p.z);
    float s = sin(k * p.z);
    return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

vec3 opElongate(vec3 p, vec3 h) {
    return p - clamp(p, -h, h);
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