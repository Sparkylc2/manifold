#version 330

in vec2 fragTexCoord;
in vec4 fragColor;

out vec4 finalColor;

void main() {
    vec2 p = fragTexCoord * 2.0 - 1.0; // map [0,1] to [-1,1]
    float dist = dot(p, p);

    if (dist > 1.0) discard;

    float alpha = 1.0 - smoothstep(0.7, 1.0, dist);
    finalColor = vec4(fragColor.rgb, fragColor.a * alpha);
}
