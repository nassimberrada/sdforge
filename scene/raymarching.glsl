float Scene(in vec3 p);
float raymarch(in vec3 ro, in vec3 rd) 
{
  float t = 0.0;
  for (int i = 0; i < 500; ++i)
  {
      vec3 p = ro + rd * t;
      float d = Scene(p);
      if (d < 0.001)  return t;
      t += d;
      if (t > 200.0)  break;
  }
  return -1.0;
}
vec3 estimateNormal(vec3 p) 
{
  float eps = 0.001;
  vec2 e = vec2(1.0, -1.0) * 0.5773 * eps;
  return normalize(vec3(
    Scene(p + vec3(e.x, e.y, e.y)) - Scene(p + vec3(e.y, e.y, e.y)),
    Scene(p + vec3(e.y, e.x, e.y)) - Scene(p + vec3(e.y, e.y, e.y)),
    Scene(p + vec3(e.y, e.y, e.x)) - Scene(p + vec3(e.y, e.y, e.y))
  ));
}