vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute( permute( permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
        + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
        + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
}

vec3 snoiseVec3( vec3 x ){
  float s  = snoise(vec3( x ));
  float s1 = snoise(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
  float s2 = snoise(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
  return vec3( s , s1 , s2 );
}

vec4 opRound(vec4 res, float r) {
    res.x -= r;
    return res;
}

vec4 opShell(vec4 res, float thickness) {
    res.x = abs(res.x) - thickness;
    return res;
}

vec4 opDisplace(vec4 res, float displacement) {
    res.x += displacement;
    return res;
}

vec4 opExtrude(vec4 res, vec3 p, float h) {
    vec2 w = vec2(res.x, abs(p.z) - h);
    res.x = min(max(w.x, w.y), 0.0) + length(max(w, 0.0));
    return res;
}

vec3 opTranslate(vec3 p, vec3 offset) {
    return p - offset;
}

vec3 opScale(vec3 p, vec3 factor) {
    return p / factor;
}

vec3 opRotateX(vec3 p, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return vec3(p.x, p.y * c + p.z * s, -p.y * s + p.z * c);
}

vec3 opRotateY(vec3 p, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return vec3(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
}

vec3 opRotateZ(vec3 p, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return vec3(p.x * c + p.y * s, -p.x * s + p.y * c, p.z);
}

vec3 opRotateAxis(vec3 p, vec3 k, float theta) {
    float c = cos(-theta);
    float s = sin(-theta);
    return p * c + cross(k, p) * s + k * dot(k, p) * (1.0 - c);
}

vec3 opOrientX(vec3 p) { return p.zyx; }
vec3 opOrientY(vec3 p) { return p.xzy; }
vec3 opOrientZ(vec3 p) { return p; }

vec3 opTwist(vec3 p, float strength) {
    float c = cos(-strength * p.y);
    float s = sin(-strength * p.y);
    mat2 m = mat2(c, -s, s, c);
    p.xz = m * p.xz;
    return p;
}

vec3 opBendX(vec3 p, float k) {
    float c = cos(k * p.x);
    float s = sin(k * p.x);
    return vec3(p.x, c * p.y + s * p.z, -s * p.y + c * p.z);
}

vec3 opBendY(vec3 p, float k) {
    float c = cos(k * p.y);
    float s = sin(k * p.y);
    return vec3(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

vec3 opBendZ(vec3 p, float k) {
    float c = cos(k * p.z);
    float s = sin(k * p.z);
    return vec3(c * p.x + s * p.y, -s * p.x + c * p.y, p.z);
}

vec3 opWarp(vec3 p, float freq, float strength) {
    return p + snoiseVec3(p * freq) * strength;
}

vec3 opRepeat(vec3 p, vec3 c) {
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