#pragma once

#include <algorithm>

#include <manifold/fluid/field_2d.h>
#include <manifold/fluid/fluid_solver.h> // Interp

// shared field interpolation
namespace manifold::Fluid {

// simple linear interpolation
inline double bilerp(const Field2D &f, double x, double y) {

    const int W = (int)f.m_W, H = (int)f.m_H;

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

// catmull-com through p1,p2 with tangents from p0,p3
inline double catmull_rom(double p0, double p1, double p2, double p3,
                          double x) {
    return p1 + 0.5 * x *
                    ((p2 - p0) + x * (2 * p0 - 5 * p1 + 4 * p2 - p3 +
                                      x * (3 * (p1 - p2) + p3 - p0)));
}

// monotone catmull-rom
// endpoint slopes clamped to the sign of dq so the interpolant can't overshoot
// (Fedkiw "sharper interpolation", eq. 5.9-5.11 from the Bridson notes)
inline double catmull_rom_monotone(double q0, double q1, double q2, double q3,
                                   double x) {
    double d1 = 0.5 * (q2 - q0);
    double d2 = 0.5 * (q3 - q1);
    const double dq = q2 - q1;
    if (dq * d1 < 0.0)
        d1 = 0.0;
    if (dq * d2 < 0.0)
        d2 = 0.0;
    const double a2 = 3.0 * dq - 2.0 * d1 - d2;
    const double a3 = -2.0 * dq + d1 + d2;
    return q1 + d1 * x + a2 * x * x + a3 * x * x * x;
}

// bicubic sample with a clamped 4x4 stencil at the borders
// `monotone` picks the kernel above
inline double bicubic(const Field2D &f, double cx, double cy, bool monotone) {
    const int W = (int)f.m_W, H = (int)f.m_H;
    cx = std::clamp(cx, 0.0, (double)W - 1.0);
    cy = std::clamp(cy, 0.0, (double)H - 1.0);

    const int i0 = (int)cx, j0 = (int)cy;
    const double s = cx - i0, t = cy - j0;

    auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
    auto k = monotone ? catmull_rom_monotone : catmull_rom;
    auto row = [&](int j) {
        return k(f(cl(i0 - 1, W - 1), j), f(cl(i0, W - 1), j),
                 f(cl(i0 + 1, W - 1), j), f(cl(i0 + 2, W - 1), j), s);
    };
    return k(row(cl(j0 - 1, H - 1)), row(cl(j0, H - 1)), row(cl(j0 + 1, H - 1)),
             row(cl(j0 + 2, H - 1)), t);
}

// fraction of a segment with endpoint sdf values pa, pb that lies in fluid.
// this is what lets a surface be seen between grid points: a face half covered
// by a body reports 0.5 instead of snapping to all-fluid or all-solid
inline double fluid_fraction(double pa, double pb) {
    if (pa >= 0.0 && pb >= 0.0)
        return 1.0;
    if (pa < 0.0 && pb < 0.0)
        return 0.0;
    return (pa >= 0.0) ? pa / (pa - pb) : pb / (pb - pa);
}

// sample with the Interp enum
// Linear -> bilinear, Cubic -> bicubic
// defaults to the monotone kernel (monotone=false uses plain catmull-rom)
inline double sample(const Field2D &f, double cx, double cy, Interp interp,
                     bool monotone = true) {
    if (interp == Interp::Linear) {
        return bilerp(f, cx, cy);
    }
    return bicubic(f, cx, cy, monotone);
}

} // namespace manifold::Fluid
