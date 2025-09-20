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

float sdCone( in vec3 p, in vec2 c ) // c.x=height, c.y=radius
{
    vec2 q = vec2( length(p.xz), p.y );
    vec2 w = vec2( c.y, c.x );    
    vec2 a = q - w*clamp( dot(q,w)/dot(w,w), 0.0, 1.0 );
    vec2 b = q - vec2( 0.0, clamp( q.y, 0.0, c.x ) );
    float k = sign( c.y );
    float d = min(dot( a, a ),dot( b, b ));
    float s = max( k*(q.x*w.y-q.y*w.x),k*(q.y-c.x) );
    return sqrt(d)*sign(s);
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

float sdBoxFrame( vec3 p, vec3 b, float e )
{
    p = abs(p)-b;
    vec3 q = abs(p+e)-e;
    return min(min(
        length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
        length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
        length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

float sdCappedTorus(vec3 p, vec2 sc, float ra, float rb)
{
    p.x = abs(p.x);
    float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
    return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}

float sdLink(vec3 p, float le, float r1, float r2)
{
    vec3 q = vec3( p.x, max(abs(p.y)-le,0.0), p.z );
    return length(vec2(length(q.xy)-r1,q.z)) - r2;
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

float sdRoundedCylinder( vec3 p, float ra, float rb, float h )
{
    vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
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

float sdRoundCone( vec3 p, float r1, float r2, float h )
{
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    vec2 q = vec2( length(p.xz), p.y );
    float k = dot(q,vec2(-b,a));
    if( k<0.0 ) return length(q) - r1;
    if( k>a*h ) return length(q-vec2(0.0,h)) - r2;
    return dot(q, vec2(a,b) ) - r1;
}

float sdPyramid( vec3 p, float h )
{
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


// --- 2D Primitives ---

float sdCircle(in vec2 p, in float r) {
    return length(p) - r;
}

float sdRectangle(in vec2 p, in vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}