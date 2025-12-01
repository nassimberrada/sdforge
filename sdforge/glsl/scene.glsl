vec4 raymarch(in vec3 ro, in vec3 rd) {
  float t = 0.0;  
  for (int i = 0; i < 500; ++i) {
      vec3 p = ro + rd * t;
      vec4 res = Scene(p);
      float d = res.x;      
      if (d < 0.001) return vec4(t, res.y, float(i), res.w); 
      t += d * 0.3;      
      if (t > 100.0) break;
  }
  return vec4(-1.0, -1.0, 500.0, 0.0); 
}

vec3 estimateNormal(vec3 p) {
  float eps = 0.002;
  vec2 e = vec2(1.0, -1.0) * 0.5773 * eps;
  return normalize(
    e.xyy * Scene(p + e.xyy).x +
    e.yyx * Scene(p + e.yyx).x +
    e.yxy * Scene(p + e.yxy).x +
    e.xxx * Scene(p + e.xxx).x
  );
}

vec3 getRayDir(vec2 st, vec3 ro, vec3 lookAt, float zoom) {
  vec3 f = normalize(lookAt - ro);
  vec3 r = normalize(cross(vec3(0,1,0), f));
  vec3 u = cross(f, r);
  return normalize(st.x * r + st.y * u + zoom * f);
}

void cameraStatic(in vec2 st, in vec3 pos, in vec3 target, in float zoom, out vec3 ro, out vec3 rd) {
    ro = pos;
    rd = getRayDir(st, ro, target, zoom);
}

void cameraOrbit(in vec2 st, in vec2 mouse, in vec2 resolution, in float zoom, out vec3 ro, out vec3 rd) {
    vec2 mouse_norm = mouse / resolution;
    float yaw = (mouse_norm.x - 0.5) * 6.28;
    float pitch = (mouse_norm.y - 0.5) * 3.14;
    pitch = clamp(pitch, -1.5, 1.5);

    float dist = 5.0;
    ro.x = dist * cos(pitch) * sin(yaw);
    ro.y = dist * sin(pitch);
    ro.z = dist * cos(pitch) * cos(yaw);
    
    rd = getRayDir(st, ro, vec3(0.0), zoom);
}

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