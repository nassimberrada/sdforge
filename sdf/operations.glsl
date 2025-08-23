float sMin(float a, float b, float k) 
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
float sUnion( float d1, float d2, float k ) 
{
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}
float sIntersect( float d1, float d2, float k ) 
{
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k*h*(1.0-h);
}
float sDifference( float d1, float d2, float k ) 
{
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h);
}
void opTwist( inout vec3 p, float k )
{
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    mat2  m = mat2(c,-s,s,c);
    p.xz = m*p.xz;
}
vec3 rotateY(vec3 vector, float theta) {
  float c =  cos(theta);
  float s =  sin(theta);
  return vec3(
      vector.x *  c + vector.z *  s,
      vector.y,
      -vector.x *  s + vector.z *  c
  );
}
vec3 rotateX(vec3 vector, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return vec3(
      vector.x,
      vector.y *  c - vector.z *  s,
      vector.y *  s + vector.z *  c
  );
}
vec3 rotateZ(vec3 vector, float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return vec3(
      vector.x *  c - vector.y *  s,
      vector.x *  s + vector.y *  c,
      vector.z
  );
}
int repeat(inout float value, float range) {
  float halfRange = range / 2.0;
  float index = floor((value + halfRange) / range);
  value = mod(value + halfRange, range) - halfRange;
  return int(index);
}
int repeatX(inout vec3 point, float range) {
  float halfRange = range / 2.0;
  float index = floor((point.x + halfRange) / range);
  point.x = mod((point.x + halfRange), range) - halfRange;
  return int(index);
}
int repeatY(inout vec3 point, float range) {
  float halfRange = range / 2.0;
  float index = floor((point.y + halfRange) / range);
  point.y = mod((point.y + halfRange), range) - halfRange;
  return int(index);
}
int repeatZ(inout vec3 point, float range) {
  float halfRange = range / 2.0;
  float index = floor((point.z + halfRange) / range);
  point.z = mod((point.z + halfRange), range) - halfRange;
  return int(index);
}
int repeatPolarX(inout vec3 point, float range, float width) {
  float angle = atan(point.y, point.z);
  float radius = length(point.yz);
  int index = repeat(angle, range);
  point.yz = radius * vec2(sin(angle), cos(angle));
  point.y -= width;
  return int(index);
}
int repeatPolarY(inout vec3 point, float range, float width) {
  float angle = atan(point.x, point.z);
  float radius = length(point.xz);
  int index = repeat(angle, range);
  point.xz = radius * vec2(sin(angle), cos(angle));
  point.z -= width;
  return int(index);
}
int repeatPolarZ(inout vec3 point, float range, float width) {
  float angle = atan(point.x, point.y);
  float radius = length(point.xy);
  int index = repeat(angle, range);
  point.xy = radius * vec2(sin(angle), cos(angle));
  point.x -= width;
  return int(index);
}
float mirror (inout float axis, float dist) {
	float s = sign(axis);
	axis = abs(axis)-dist;
	return s;
}
vec2 mirrorDiag (inout vec2 point, vec2 dist) {
	vec2 s = sign(point);
	mirror(point.x, dist.x);
	mirror(point.y, dist.y);
	if (point.y > point.x)
		point.xy = point.yx;
	return s;
}