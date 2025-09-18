// --- Transformations ---

vec4 opTwist( in vec4 sdf_result, inout vec3 p, float k )
{
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    mat2  m = mat2(c,-s,s,c);
    p.xz = m*p.xz;

    return sdf_result;
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


// --- Domain Repetition ---

vec3 opRepeat(vec3 p, vec3 c)
{
    return mod(p + 0.5 * c, c) - 0.5 * c;
}

// --- Mirroring ---

vec3 opMirror(vec3 p, vec3 a) {
    if (a.x > 0.5) p.x = abs(p.x);
    if (a.y > 0.5) p.y = abs(p.y);
    if (a.z > 0.5) p.z = abs(p.z);
    return p;
}