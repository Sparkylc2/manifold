#version 330
in vec2 fragTexCoord; // .y perp across the ribbon, 0..1
in vec4 fragColor; // per vertex for age gradient

out vec4 finalColor;

void main() {
    float d = abs(fragTexCoord.y - 0.5) * 2.0; // 0 at core and 1 at edge

    float aa = clamp(fwidth(d), 0.01, 0.5);

    float edge = 1.0 - smoothstep(1.0 - aa, 1.0, d);
    float core = 1.0 - d * d; // soft falloff across width
    finalColor = vec4(fragColor.rgb, fragColor.a * core * edge);
}
