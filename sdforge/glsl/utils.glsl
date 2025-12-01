vec3 debugNormals(vec3 normal) {
    return normal * 0.5 + 0.5;
}

vec3 debugSteps(float steps, float max_steps) {
    float x = clamp(steps / max_steps, 0.0, 1.0);
    return vec3(x, 1.0 - x, 1.0);
}

vec3 debugDistanceField(float d) {
    vec3 col = (d > 0.0) ? vec3(1.0, 0.65, 0.1) : vec3(0.1, 0.4, 0.8);
    col *= 1.0 - exp(-3.0 * abs(d));
    col *= 0.8 + 0.2 * cos(120.0 * d);
    col = mix(col, vec3(1.0), 1.0 - smoothstep(0.0, 0.015, abs(d)));

    return col;
}