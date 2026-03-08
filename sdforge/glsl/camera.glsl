vec3 getRayDir(vec2 st, vec3 ro, vec3 lookAt, float zoom) {
  vec3 f = normalize(lookAt - ro);
  vec3 r = normalize(cross(vec3(0,1,0), f));
  vec3 u = cross(f, r);
  return normalize(st.x * r + st.y * u + zoom * f);
}

void cameraOrbit(in vec2 st, in vec2 mouse_offset, in vec2 resolution, in vec3 base_pos, in vec3 target, in float zoom, out vec3 ro, out vec3 rd) {
    vec3 dir = base_pos - target;
    float dist = length(dir);
    
    // Calculate initial base angles from the configured position
    float base_yaw = atan(dir.x, dir.z);
    float base_pitch = asin(clamp(dir.y / dist, -1.0, 1.0));
    
    // Apply mouse drag offsets
    vec2 delta = mouse_offset / resolution;
    float yaw = base_yaw + delta.x * 6.28;
    float pitch = base_pitch + delta.y * 3.14;
    pitch = clamp(pitch, -1.5, 1.5); // Prevent flipping upside down
    
    ro = target + vec3(
        dist * cos(pitch) * sin(yaw),
        dist * sin(pitch),
        dist * cos(pitch) * cos(yaw)
    );
    
    rd = getRayDir(st, ro, target, zoom);
}

void cameraOrbitOrtho(in vec2 st, in vec2 mouse_offset, in vec2 resolution, in vec3 base_pos, in vec3 target, in float zoom, out vec3 ro, out vec3 rd) {
    vec3 dir = base_pos - target;
    float dist = length(dir);
    
    float base_yaw = atan(dir.x, dir.z);
    float base_pitch = asin(clamp(dir.y / dist, -1.0, 1.0));
    
    vec2 delta = mouse_offset / resolution;
    float yaw = base_yaw + delta.x * 6.28;
    float pitch = base_pitch + delta.y * 3.14;
    pitch = clamp(pitch, -1.5, 1.5);
    
    vec3 pos = target + vec3(
        dist * cos(pitch) * sin(yaw),
        dist * sin(pitch),
        dist * cos(pitch) * cos(yaw)
    );
    
    vec3 f = normalize(target - pos);
    vec3 up = vec3(0.0, 1.0, 0.0);
    if (abs(dot(f, up)) > 0.999) { up = vec3(0.0, 0.0, 1.0); }
    vec3 r = normalize(cross(up, f));
    vec3 u = cross(f, r);
    
    float ortho_scale = 5.0 / max(zoom, 0.001);
    ro = pos + (st.x * r + st.y * u) * ortho_scale - f * 10.0;
    rd = f;
}