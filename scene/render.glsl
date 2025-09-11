vec4 render(vec2 uv, vec2 u_resolution, float u_time, vec4 u_mouse, float zoom, vec3 base_color) {
    vec2 st = (2.0 * uv - 1.0) * vec2(u_resolution.x / u_resolution.y, 1.0);
    vec3 ro, rd;
    cameraOrbit(st, u_mouse.xy, u_resolution, zoom, ro, rd);

    vec3 col = vec3(0.0);
    float t = raymarch(ro, rd);
    if (t > 0.0) {
        vec3 pos = ro + t * rd;
        vec3 normal = estimateNormal(pos);
        vec3 lightDir = normalize(vec3(0.2, 0.2, 2.0));
        float diff = max(dot(normal, lightDir), 0.0);
        col = base_color * diff;
    } else {
      col = vec3(1.0);
    }

    return vec4(col, 1.0);
}