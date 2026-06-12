#version 330

in vec2 fragTexCoord;
in vec4 fragColor;

out vec4 finalColor;

uniform sampler2D texture0;
uniform vec2 resolution;

void main() {
    vec2 texel = 1.0 / resolution;

    // sample neighborhood
    vec3 rgbNW = texture(texture0, fragTexCoord + vec2(-1.0, -1.0) * texel).rgb;
    vec3 rgbNE = texture(texture0, fragTexCoord + vec2( 1.0, -1.0) * texel).rgb;
    vec3 rgbSW = texture(texture0, fragTexCoord + vec2(-1.0,  1.0) * texel).rgb;
    vec3 rgbSE = texture(texture0, fragTexCoord + vec2( 1.0,  1.0) * texel).rgb;
    vec3 rgbM  = texture(texture0, fragTexCoord).rgb;

    // luminance weights (rec. 709)
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    // edge direction
    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * (1.0 / 8.0),
        1.0 / 128.0
    );
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

    dir = min(vec2(8.0), max(vec2(-8.0), dir * rcpDirMin)) * texel;

    // two-tap blend
    vec3 rgbA = 0.5 * (
        texture(texture0, fragTexCoord + dir * (1.0 / 3.0 - 0.5)).rgb +
        texture(texture0, fragTexCoord + dir * (2.0 / 3.0 - 0.5)).rgb
    );

    // four-tap blend
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(texture0, fragTexCoord + dir * -0.5).rgb +
        texture(texture0, fragTexCoord + dir *  0.5).rgb
    );

    float lumaB = dot(rgbB, luma);

    if (lumaB < lumaMin || lumaB > lumaMax) {
        finalColor = vec4(rgbA, 1.0);
    } else {
        finalColor = vec4(rgbB, 1.0);
    }
}
