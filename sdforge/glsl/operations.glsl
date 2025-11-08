// --- Standard Boolean Operations ---

vec4 opU(vec4 a, vec4 b) {
    return (a.x < b.x) ? a : b;
}

vec4 opI(vec4 a, vec4 b) {
    return (a.x > b.x) ? a : b;
}

vec4 opS(vec4 a, vec4 b) {
    // Subtraction is intersection with the inverted second object
    return opI(a, vec4(-b.x, b.y, b.z, b.w));
}


// --- Smooth Boolean Operations ---

vec4 sUnion(vec4 a, vec4 b, float k )
{
    float h = clamp( 0.5 + 0.5*(b.x-a.x)/k, 0.0, 1.0 );
    float dist = mix( b.x, a.x, h ) - k*h*(1.0-h);
    // The material of the object with the smaller distance is chosen
    return (a.x < b.x) ? vec4(dist, a.y, a.z, a.w) : vec4(dist, b.y, b.z, b.w);
}

vec4 sIntersect(vec4 a, vec4 b, float k )
{
    float h = clamp( 0.5 - 0.5*(b.x-a.x)/k, 0.0, 1.0 );
    float dist = mix( b.x, a.x, h ) + k*h*(1.0-h);
    // The material of the object with the larger distance is chosen
    return (a.x > b.x) ? vec4(dist, a.y, a.z, a.w) : vec4(dist, b.y, b.z, b.w);
}

vec4 sDifference(vec4 a, vec4 b, float k )
{
    float h = clamp( 0.5 - 0.5*(b.x+a.x)/k, 0.0, 1.0 );
    float dist = mix( a.x, -b.x, h ) + k*h*(1.0-h);
    // The material of the first object is always used for subtraction
    return vec4(dist, a.y, a.z, a.w);
}