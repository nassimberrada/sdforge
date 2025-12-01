vec4 opU(vec4 a, vec4 b) {
    return (a.x < b.x) ? a : b;
}

vec4 opI(vec4 a, vec4 b) {
    return (a.x > b.x) ? a : b;
}

vec4 opS(vec4 a, vec4 b) {
    return opI(a, vec4(-b.x, b.y, b.z, b.w));
}

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

vec4 cUnion(vec4 a, vec4 b, float k)
{
    float h = clamp( 0.5 + 0.5*(b.x-a.x)/k, 0.0, 1.0 );
    float dist = mix( b.x, a.x, h );
    return (a.x < b.x) ? vec4(dist, a.y, a.z, a.w) : vec4(dist, b.y, b.z, b.w);
}

vec4 cIntersect(vec4 a, vec4 b, float k)
{
    float h = clamp( 0.5 - 0.5*(b.x-a.x)/k, 0.0, 1.0 );
    float dist = mix( b.x, a.x, h );
    return (a.x > b.x) ? vec4(dist, a.y, a.z, a.w) : vec4(dist, b.y, b.z, b.w);
}

vec4 cDifference(vec4 a, vec4 b, float k)
{
    float h = clamp( 0.5 - 0.5*(b.x+a.x)/k, 0.0, 1.0 );
    float dist = mix( a.x, -b.x, h );
    return vec4(dist, a.y, a.z, a.w);
}

vec4 opMorph(vec4 a, vec4 b, float t) {
    float factor = clamp(t, 0.0, 1.0);
    float dist = mix(a.x, b.x, factor);
    return (factor < 0.5) ? vec4(dist, a.y, a.z, a.w) : vec4(dist, b.y, b.z, b.w);
}