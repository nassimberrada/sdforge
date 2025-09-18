vec4 Scene(in vec3 p);

vec4 raymarch(in vec3 ro, in vec3 rd) {
  float t = 0.0;
  for (int i = 0; i < 100; ++i) {
      vec3 p = ro + rd * t;
      vec4 res = Scene(p);
      float d = res.x;
      if (d < 0.001) return vec4(t, res.y, res.z, res.w);
      t += d;
      if (t > 100.0) break;
  }
  return vec4(-1.0);
}

vec3 estimateNormal(vec3 p) {
  float eps = 0.001;
  vec2 e = vec2(1.0, -1.0) * 0.5773 * eps;
  return normalize(
    e.xyy * Scene(p + e.xyy).x +
    e.yyx * Scene(p + e.yyx).x +
    e.yxy * Scene(p + e.yxy).x +
    e.xxx * Scene(p + e.xxx).x
  );
}