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

float sdTorus(in vec3 p, in vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdCapsule(in vec3 p, in vec3 a, in vec3 b, in float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdCone( in vec3 p, in vec2 c ) {
    vec2 q = vec2( length(p.xz), p.y );
    vec2 w = vec2( c.y, c.x );    
    vec2 a = q - w*clamp( dot(q,w)/dot(w,w), 0.0, 1.0 );
    vec2 b = q - vec2( 0.0, clamp( q.y, 0.0, c.x ) );
    float k = sign( c.y );
    float d = min(dot( a, a ),dot( b, b ));
    float s = max( k*(q.x*w.y-q.y*w.x),k*(q.y-c.x) );
    return sqrt(d)*sign(s);
}

float sdPlane(in vec3 p, in vec4 n) {
    return dot(p, n.xyz) + n.w;
}

float sdHexPrism(in vec3 p, in vec2 h) {
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

float sdEllipsoid(in vec3 p, in vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float sdCylinder(vec3 p, vec2 h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdCappedCylinder( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 ba = b - a;
    vec3 pa = p - a;
    float baba = dot(ba,ba);
    float paba = dot(pa,ba);
    float x = length(pa*baba-ba*paba) - r*baba;
    float y = abs(paba-baba*0.5)-baba*0.5;
    float x2 = x*x;
    float y2 = y*y*baba;
    float d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
    return sign(d)*sqrt(abs(d))/baba;
}

float sdCappedCone( vec3 p, float h, float r1, float r2 )
{
    vec2 q = vec2( length(p.xz), p.y );
    vec2 k1 = vec2(r2,h);
    vec2 k2 = vec2(r2-r1,2.0*h);
    vec2 ca = vec2(q.x-min(q.x,(q.y<0.0)?r1:r2), abs(q.y)-h);
    vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot(k2,k2), 0.0, 1.0 );
    float s = (cb.x<0.0 && ca.y<0.0) ? -1.0 : 1.0;
    return s*sqrt( min(dot(ca,ca),dot(cb,cb)) );
}

float sdPyramid( vec3 p, float h )
{
    p.y += h * 0.5;    
    float m2 = h*h + 0.25;
    p.xz = abs(p.xz);
    p.xz = (p.z>p.x) ? p.zx : p.xz;
    p.xz -= 0.5;
    vec3 q = vec3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);
    float s = max(-q.x,0.0);
    float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );
    float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
    float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);
    float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);
    return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));
}

int solveCubic(float a, float b, float c, out float r[3])
{
    float p = b - a*a / 3.0;
    float q = 2.0*a*a*a / 27.0 - a*b / 3.0 + c;
    float p3 = p*p*p;
    float d = q*q / 4.0 + p3 / 27.0;
    if (d > 0.0) {
        vec2 x = (vec2(1.0, -1.0) * sqrt(d) - q / 2.0);
        x = sign(x) * pow(abs(x), vec2(1.0 / 3.0));
        r[0] = x.x + x.y - a / 3.0;
        return 1;
    }
    float v = acos(-sqrt(-27.0 / p3) * q / 2.0) / 3.0;
    float m = cos(v);
    float n = sin(v) * 1.732050808;
    vec3 t = vec3(m + m, -n - m, n - m) * sqrt(-p / 3.0) - a / 3.0;
    r[0] = t.x; r[1] = t.y; r[2] = t.z;
    return 3;
}

float sdBezier(vec3 pos, vec3 A, vec3 B, vec3 C, float r)
{
    vec3 a = B - A;
    vec3 b = A - 2.0*B + C;
    vec3 c = a * 2.0;
    vec3 d = A - pos;
    float kk = 1.0 / dot(b,b);
    float kx = kk * dot(a,b);
    float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
    float kz = kk * dot(d,a);
    float res = 0.0;
    float roots[3];
    int n = solveCubic(3.0*kx, 3.0*ky, kz, roots);
    float t = clamp(roots[0], 0.0, 1.0);
    vec3 q = d + (c + b*t)*t;
    res = dot(q,q);
    if(n>1) {
        t = clamp(roots[1], 0.0, 1.0);
        q = d + (c + b*t)*t;
        res = min(res, dot(q,q));
        t = clamp(roots[2], 0.0, 1.0);
        q = d + (c + b*t)*t;
        res = min(res, dot(q,q));
    }
    return sqrt(res) - r;
}

float sdCircle(in vec2 p, in float r) {
    return length(p) - r;
}

float sdRectangle(in vec2 p, in vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}

float sdEquilateralTriangle(in vec2 p, in float r) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x+k*p.y > 0.0 ) p = vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p)*sign(p.y);
}

float sdTrapezoid(in vec2 p, in float r1, in float r2, in float he) {
    vec2 k1 = vec2(r2,he);
    vec2 k2 = vec2(r2-r1,2.0*he);
    p.x = abs(p.x);
    vec2 ca = vec2(p.x-min(p.x,(p.y<0.0)?r1:r2), abs(p.y)-he);
    vec2 cb = p - k1 + k2*clamp( dot(k1-p,k2)/dot(k2,k2), 0.0, 1.0 );
    float s = (cb.x<0.0 && ca.y<0.0) ? -1.0 : 1.0;
    return s*sqrt( min(dot(ca,ca),dot(cb,cb)) );
}