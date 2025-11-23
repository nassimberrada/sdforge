vec4 raymarch(in vec3 ro, in vec3 rd) {
  float t = 0.0;
  
  // Significant increase in max steps to handle the slowdown required for warping
  for (int i = 0; i < 500; ++i) {
      vec3 p = ro + rd * t;
      vec4 res = Scene(p);
      float d = res.x;
      
      // Surface hit threshold
      if (d < 0.001) return vec4(t, res.y, float(i), res.w); 
      
      // Conservative stepping:
      // Domain warping can compress space by a factor of (1 + strength * freq).
      // A factor of 0.3 is generally safe for moderate warping.
      // Without this, the ray overshoots the surface, entering the negative distance field,
      // which causes normal calculation to fail (black pixels).
      t += d * 0.3;
      
      // Far plane clip
      if (t > 100.0) break;
  }
  
  // Miss
  return vec4(-1.0, -1.0, 500.0, 0.0); 
}

vec3 estimateNormal(vec3 p) {
  // Slightly larger epsilon for finite difference helps smooth out 
  // high-frequency noise artifacts in the normal calculation
  float eps = 0.002;
  vec2 e = vec2(1.0, -1.0) * 0.5773 * eps;
  return normalize(
    e.xyy * Scene(p + e.xyy).x +
    e.yyx * Scene(p + e.yyx).x +
    e.yxy * Scene(p + e.yxy).x +
    e.xxx * Scene(p + e.xxx).x
  );
}