// --- Debug Visualization Functions ---

// Visualizes a normal vector as an RGB color.
// Maps (x, y, z) components from [-1, 1] to [0, 1].
vec3 debugNormals(vec3 normal) {
    return normal * 0.5 + 0.5;
}

// Visualizes the number of raymarching steps using a color gradient.
// Blue (few steps) -> Red (many steps).
vec3 debugSteps(float steps, float max_steps) {
    float x = clamp(steps / max_steps, 0.0, 1.0);
    // A simple gradient from blue to red
    return vec3(x, 1.0 - x, 1.0);
}

// Visualizes a signed distance field value.
// Inside: Blue. Outside: Orange.
// Isolines: White curves at regular intervals.
vec3 debugDistanceField(float d) {
    // Basic coloring: Blue for inside (-), Orange for outside (+)
    vec3 col = (d > 0.0) ? vec3(1.0, 0.65, 0.1) : vec3(0.1, 0.4, 0.8);

    // Darken towards the zero-crossing to emphasize the surface
    col *= 1.0 - exp(-3.0 * abs(d));

    // Add periodic rings (isolines)
    col *= 0.8 + 0.2 * cos(120.0 * d);

    // Draw a sharp white line at the surface (d=0)
    col = mix(col, vec3(1.0), 1.0 - smoothstep(0.0, 0.015, abs(d)));

    return col;
}