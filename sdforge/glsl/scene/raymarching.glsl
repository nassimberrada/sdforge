float Scene(in vec3 p); // Forward declaration

float raymarch(in vec3 ro, in vec3 rd) {
  float t = 0.0;
  for (int i = 0; i < 100; ++i) {
      vec3 p = ro + rd * t;
      float d = Scene(p);
      if (d < 0.001) return t;
      t += d;
      if (t > 100.0) break;
  }
  return -1.0;
}

vec3 estimateNormal(vec3 p) {
  float eps = 0.001;
  vec2 e = vec2(1.0, -1.0) * 0.5773 * eps;
  return normalize(
    e.xyy * Scene(p + e.xyy) +
    e.yyx * Scene(p + e.yyx) +
    e.yxy * Scene(p + e.yxy) +
    e.xxx * Scene(p + e.xxx)
  );
}