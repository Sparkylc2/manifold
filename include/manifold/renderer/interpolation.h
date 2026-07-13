#pragma once

#include <algorithm>

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

} // namespace manifold::Rendering
