float softShadow(vec3 ro, vec3 rd) 
{
    float res = 1.0;
    float t = 0.01;
    for (int i = 0; i < 32; i++) {
        float h = Scene(ro + rd * t);
        if (h < 0.001) return 0.0;  // in shadow
        res = min(res, 10.0 * h / t);  // softness factor
        t += h;
        if (t > 5.0) break;
    }
    return clamp(res, 0.0, 1.0);
}
float ambientOcclusion(vec3 p, vec3 n) 
{
  float ao = 0.0;
  float sca = 1.0;
  for (int i = 1; i <= 5; i++) {
      float h = 0.01 * float(i);
      float d = Scene(p + n * h);
      ao += (h - d) * sca;
      sca *= 0.7;
  }
  return clamp(1.0 - ao, 0.0, 1.0);
}