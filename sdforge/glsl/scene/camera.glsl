vec3 getRayDir(vec2 st, vec3 ro, vec3 lookAt, float zoom) {
  vec3 f = normalize(lookAt - ro);
  vec3 r = normalize(cross(vec3(0,1,0), f));
  vec3 u = cross(f, r);
  return normalize(st.x * r + st.y * u + 1.5 * f / zoom);
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