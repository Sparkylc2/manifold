#pragma once

#include <algorithm>
#include <cmath>

#include <manifold/renderer/theme.h>

// general-purpose interpolation (field-specific kernels live in fluid/)
namespace manifold::Rendering {

inline double lerp(double a, double b, double t) { return a + (b - a) * t; }

// bilinear sample over a clamped grid via an f(i,j) -> double accessor
template <typename F>
inline double bilerp(F &&f, int W, int H, double x, double y) {
    x = std::clamp(x, 0.0, (double)W - 1.0);
    y = std::clamp(y, 0.0, (double)H - 1.0);

    const int i0 = (int)x, j0 = (int)y;
    const int i1 = std::min(i0 + 1, W - 1);
    const int j1 = std::min(j0 + 1, H - 1);
    const double sx1 = x - i0, sx0 = 1.0 - sx1;
    const double sy1 = y - j0, sy0 = 1.0 - sy1;

    return sx0 * (sy0 * f(i0, j0) + sy1 * f(i0, j1)) +
           sx1 * (sy0 * f(i1, j0) + sy1 * f(i1, j1));
}

inline Color color_lerp(Color a, Color b, double f) {
    return {(unsigned char)(a.r + f * ((double)b.r - a.r)),
            (unsigned char)(a.g + f * ((double)b.g - a.g)),
            (unsigned char)(a.b + f * ((double)b.b - a.b)),
            (unsigned char)(a.a + f * ((double)b.a - a.a))};
}

// diverging ramp about mid: t>0 -> mid..hi, t<0 -> mid..lo
inline Color diverging_ramp(Color mid, Color lo, Color hi, double t) {
    t = std::clamp(t, -1.0, 1.0);
    return t >= 0 ? color_lerp(mid, hi, t) : color_lerp(mid, lo, -t);
}

// Oklab. mixing two saturated hues in RGB runs the path through a desaturated
// grey; in Oklab the midpoint stays a colour, which is what a weighted blend of
// several accents needs to survive
struct Oklab {
    double L = 0.0, a = 0.0, b = 0.0;
};

inline double srgb_to_linear(double c) {
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

inline double linear_to_srgb(double c) {
    return c <= 0.0031308 ? 12.92 * c
                          : 1.055 * std::pow(std::max(c, 0.0), 1.0 / 2.4) - 0.055;
}

inline Oklab to_oklab(Color c) {
    const double r = srgb_to_linear(c.r / 255.0);
    const double g = srgb_to_linear(c.g / 255.0);
    const double b = srgb_to_linear(c.b / 255.0);

    const double l = std::cbrt(0.4122214708 * r + 0.5363325363 * g +
                               0.0514459929 * b);
    const double m = std::cbrt(0.2119034982 * r + 0.6806995451 * g +
                               0.1073969566 * b);
    const double s = std::cbrt(0.0883024619 * r + 0.2817188376 * g +
                               0.6299787005 * b);

    return {0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s};
}

inline Color from_oklab(const Oklab &o, unsigned char alpha = 255) {
    const double l = std::pow(o.L + 0.3963377774 * o.a + 0.2158037573 * o.b, 3);
    const double m = std::pow(o.L - 0.1055613458 * o.a - 0.0638541728 * o.b, 3);
    const double s = std::pow(o.L - 0.0894841775 * o.a - 1.2914855480 * o.b, 3);

    const double r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    const double g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    const double b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    auto q = [](double v) {
        return (unsigned char)(255.0 *
                               std::clamp(linear_to_srgb(v), 0.0, 1.0));
    };
    return {q(r), q(g), q(b), alpha};
}

inline Oklab oklab_lerp(const Oklab &x, const Oklab &y, double f) {
    f = std::clamp(f, 0.0, 1.0);
    return {lerp(x.L, y.L, f), lerp(x.a, y.a, f), lerp(x.b, y.b, f)};
}

inline Color color_lerp_oklab(Color x, Color y, double f) {
    return from_oklab(oklab_lerp(to_oklab(x), to_oklab(y), f),
                      (unsigned char)lerp(x.a, y.a, std::clamp(f, 0.0, 1.0)));
}

} // namespace manifold::Rendering
