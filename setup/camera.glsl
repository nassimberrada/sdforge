vec3 getRayDir(vec2 st, vec3 ro, vec3 lookAt) 
{
  vec3 f = normalize(lookAt - ro);
  vec3 r = normalize(cross(vec3(0,1,0), f));
  vec3 u = cross(f, r);
  return normalize(st.x * r + st.y * u + 1.5 * f);
}
void cameraFixed(in vec2 uv, inout vec3 ro, in vec3 rt, out vec3 rd) 
{
  vec3 forward = normalize(rt - ro);
  vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
  vec3 up = cross(forward, right);
  rd = normalize(uv.x * right + uv.y * up + forward);
}
void cameraOrbit(in vec2 uv, in vec2 mouse, in vec2 resolution, in float zoom, out vec3 ro, out vec3 rd) 
{
    vec2  mouseNorm = (mouse.xy / resolution) * 2.0 - 1.0;
    float sensitivity = 3.1415;
    float yaw   = mouseNorm.x * sensitivity;
    float pitch = clamp(mouseNorm.y * sensitivity * 0.5, -1.5, 1.5);

    float camera = zoom * 50.0;
    float x = camera * cos(pitch) * sin(yaw);
    float y = camera * sin(pitch);
    float z = camera * cos(pitch) * cos(yaw);
    ro = vec3(x, y, z);
    
    rd = getRayDir(uv, ro, vec3(0.0));
}
void cameraFirstPerson( in vec2 uv, in vec2 mouse, in vec2 resolution, in float time, in vec4 movement,  // x = forward, y = backward, z = left, w = right in float speed, out vec3 ro, out vec3 rd) 
{
  // Mouse look
  vec2 mouseNorm = (mouse.xy / resolution.xy) * 2.0 - 1.0;
  float sensitivity = 3.1415;

  // Camera rotation
  float yaw   = mouseNorm.x * sensitivity;
  float pitch = clamp(mouseNorm.y * sensitivity * 0.5, -1.5, 1.5);

  // Build camera direction vectors
  vec3 forward = normalize(vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)));
  vec3 right   = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
  vec3 up      = cross(forward, right);

  // Movement vector
  vec3 moveDir = vec3(0.0);
  moveDir += forward * (movement.x - movement.y);  // forward/back
  moveDir += right   * (movement.w - movement.z);  // right/left

  // Camera position
  ro = moveDir * speed * time;

  // Ray direction
  rd = normalize(uv.x * right + uv.y * up + 1.5 * forward);
}