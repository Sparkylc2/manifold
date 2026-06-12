#version 330

in vec2 fragTexCoord;
in vec4 fragColor;

out vec4 finalColor;

uniform sampler2D texture0;

void main() {
    // fragTexCoord.y encodes perpendicular distance:
    // 0.0 at outer edge, 0.5 at center, 1.0 at opposite outer edge
    float dist = abs(fragTexCoord.y - 0.5) * 2.0; // 0 at center, 1 at edge
    float alpha = 1.0 - smoothstep(0.6, 1.0, dist);

    finalColor = vec4(fragColor.rgb, fragColor.a * alpha);
}
