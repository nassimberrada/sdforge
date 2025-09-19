float softShadow(vec3 ro, vec3 rd, float softness) {
    float res = 1.0;
    float t = 0.02;
    for (int i = 0; i < 32; i++) {
        float h = Scene(ro + rd * t).x;
        if (h < 0.001) return 0.0;
        res = min(res, softness * h / t);
        t += h;
        if (t > 10.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

float ambientOcclusion(vec3 p, vec3 n, float strength) {
  float ao = 0.0;
  float sca = 1.0;
  for (int i = 0; i < 5; i++) {
      float h = 0.01 + 0.1 * float(i) / 4.0;
      float d = Scene(p + n * h).x;
      ao += -(d-h)*sca;
      sca *= 0.95;
  }
  return clamp(1.0 - strength * ao, 0.0, 1.0);
}