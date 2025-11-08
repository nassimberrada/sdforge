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