#version 330
in vec2 fragTexCoord; // .x arc along the body 0..1, .y across the ribbon 0..1
in vec4 fragColor; // per vertex for the age/tail gradient

out vec4 finalColor;

// the light sits off to one side, so the lit band lands off-centre. a highlight
// down the middle reads as a flat ribbon lit head-on; an offset one is what
// makes the eye call it a tube
const float LIGHT = -0.42;

void main() {
    float d = fragTexCoord.y * 2.0 - 1.0; // -1 and +1 at the two edges
    float ad = abs(d);

    float aa = clamp(fwidth(ad), 0.004, 0.5);
    float edge = 1.0 - smoothstep(1.0 - aa, 1.0, ad);

    // z of the normal of a unit cylinder cut across: sqrt, not the parabola
    // 1 - d*d, because the parabola falls off too early and flattens the body
    float n = sqrt(max(1.0 - ad * ad, 0.0));

    // The cross-section is shaded in ALPHA, not in rgb.
    //
    // Multiplying rgb by anything under 1 is a multiply toward BLACK, so a blob
    // handed exactly the background colour still came out with a grey rim --
    // and no choice of colour could avoid it, because every colour times 0.78
    // is a darker version of that colour. That was the dark blob.
    //
    // Modulating opacity instead leaves every pixel exactly the colour it was
    // given and lets the field underneath supply the contrast: the lit flank
    // covers the field more, the far flank lets it through. Nothing on screen
    // is ever darker than what was already there.
    float lit = clamp(0.62 * n - 0.48 * d, 0.0, 1.0);
    float body = 0.45 + 0.55 * lit;

    // set this above 0 to put some of the shading back into rgb; it will
    // darken toward black again, which is exactly what it looks like
    const float SHADE_RGB = 0.0;
    vec3 rgb = fragColor.rgb * (1.0 - SHADE_RGB * (1.0 - body));

    // PREMULTIPLIED: the merge buffer adds colour and alpha together, and the
    // resolve divides one by the other, so colour has to arrive already
    // weighted by its own coverage or overlaps come out wrong
    float a = fragColor.a * n * edge * body;
    finalColor = vec4(rgb * a, a);
}
