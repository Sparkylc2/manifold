#version 330
in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D texture0;

out vec4 finalColor;

// the knee the accumulated alpha is thresholded at. two blobs that only just
// touch each push their overlap past T0 together and come out as one contour,
// which is the whole reason for the offscreen pass; widen the gap for a softer
// join, narrow it for a harder one
const float T0 = 0.30;
const float T1 = 0.68;

void main() {
    vec4 c = texture(texture0, fragTexCoord);
    if (c.a <= 0.002)
        discard;

    // colour arrived blended against a transparent buffer, so it is
    // premultiplied by the alpha that accumulated over it
    vec3 rgb = c.rgb / max(c.a, 1e-3);

    finalColor = vec4(rgb, smoothstep(T0, T1, c.a) * fragColor.a);
}
