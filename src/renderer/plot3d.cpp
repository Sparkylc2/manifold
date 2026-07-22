#include "raylib.h"
#include <manifold/renderer/plot3d.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace manifold::Rendering {

static ::Color rl(Color c) { return {c.r, c.g, c.b, c.a}; }

static float &axis_ref(Vector3 &v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : v.z);
}
static float axis_val(const Vector3 &v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : v.z);
}

void Plot3D::Bounds::include(Vector3 p) {
    lo = Vector3Min(lo, p);
    hi = Vector3Max(hi, p);
}
Vector3 Plot3D::Bounds::centre() const { return (lo + hi) / 2; }
Vector3 Plot3D::Bounds::span() const { return hi - lo; }

void Plot3D::set_bounds(Bounds b) { m_bounds = b; }
void Plot3D::set_cube_half(double h) { m_cube_half = h; }
void Plot3D::set_scaling(Scaling s) { m_scaling = s; }

void Plot3D::fit(const std::vector<Vector3> &pts, double pad) {
    if (pts.empty())
        return;

    m_bounds.lo = {FLT_MAX, FLT_MAX, FLT_MAX};
    m_bounds.hi = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (const Vector3 &pt : pts)
        m_bounds.include(pt);

    const Vector3 vec_pad = m_bounds.span() * pad;
    m_bounds.lo -= vec_pad;
    m_bounds.hi += vec_pad;
}

Vector3 Plot3D::map(Vector3 p) const {
    if (m_scaling == Scaling::PerAxis)
        return (p - m_bounds.centre()) * 2.0 / m_bounds.span() * m_cube_half;
    if (m_scaling == Scaling::Uniform) {
        const Vector3 half_x = m_bounds.span() / 2.0;
        const double max_dim = std::max(std::max(half_x.x, half_x.y), half_x.z);
        return (p - m_bounds.centre()) * m_cube_half / max_dim;
    }
    return Vector3Zero();
}

void Plot3D::draw_box(Color edge) const {
    const float cube_half = m_cube_half;
    Vector3 corners[8];
    for (int i = 0; i < 8; ++i)
        corners[i] = {(i & 1) ? cube_half : -cube_half,
                      (i & 2) ? cube_half : -cube_half,
                      (i & 4) ? cube_half : -cube_half};

    for (int i = 0; i < 8; ++i)
        for (int b = 1; b <= 4; b <<= 1) {
            const int j = i ^ b;
            if (i < j)
                DrawLine3D(corners[i], corners[j], rl(edge));
        }
}

void Plot3D::draw_ticks(int axis, int count, Color c) const {
    if (axis < 0 || axis > 2)
        return;

    const double amin = axis_val(m_bounds.lo, axis);
    const double amax = axis_val(m_bounds.hi, axis);
    const double step = nice_step(amax - amin, count);
    if (step <= 0.0)
        return;

    const float mark = 0.05f * (float)m_cube_half;
    const int perp = (axis + 1) % 3;
    const double first = std::ceil(amin / step) * step;

    for (double v = first; v <= amax + step * 1e-6; v += step) {
        Vector3 d = m_bounds.lo;
        axis_ref(d, axis) = (float)v;
        const Vector3 a = map(d);
        Vector3 b = a;
        axis_ref(b, perp) -= mark;
        DrawLine3D(a, b, rl(c));
    }
}

void Plot3D::draw_curve(const std::vector<Vector3> &pts, Color c,
                        double tube_r) const {
    if (pts.size() < 2)
        return;

    Vector3 prev = map(pts[0]);
    for (size_t i = 1; i < pts.size(); ++i) {
        const Vector3 cur = map(pts[i]);
        if (tube_r > 0.0)
            DrawCylinderEx(prev, cur, (float)tube_r, (float)tube_r, 8, rl(c));
        else
            DrawLine3D(prev, cur, rl(c));
        prev = cur;
    }
}

void Plot3D::draw_points(const std::vector<Vector3> &pts, double r,
                         Color c) const {
    for (const Vector3 &p : pts)
        DrawSphere(map(p), (float)r, rl(c));
}

bool Plot3D::project(const Camera3D &cam, Renderer *r, Vector3 cube_pt,
                     double ox, double oy, double w, double h, int *sx,
                     int *sy) const {
    const Vector3 fwd =
        Vector3Normalize(Vector3Subtract(cam.target, cam.position));
    if (Vector3DotProduct(fwd, cube_pt - cam.position) <= 0.0f)
        return false;

    const Vector2 uv = GetWorldToScreenEx(cube_pt, cam, 1, 1);

    int x0, y0, x1, y1;
    r->world_to_screen(ox, oy + h, &x0, &y0);
    r->world_to_screen(ox + w, oy, &x1, &y1);

    *sx = (int)std::lround(x0 + uv.x * (x1 - x0));
    *sy = (int)std::lround(y0 + uv.y * (y1 - y0));
    return true;
}

void Plot3D::draw_axis_labels(Renderer *r, const Camera3D &cam, double ox,
                              double oy, double w, double h) const {
    const char *names[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; ++a) {
        Vector3 d = m_bounds.lo;
        axis_ref(d, a) = axis_val(m_bounds.hi, a);

        int sx, sy;
        if (!project(cam, r, map(d), ox, oy, w, h, &sx, &sy))
            continue;

        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s %.3g", names[a],
                      (double)axis_val(m_bounds.hi, a));
        r->draw_text(buf, sx, sy, 18, palette::text());
    }
}

double Plot3D::nice_step(double range, int target) {
    if (range <= 0.0 || target < 1)
        return 0.0;

    const double raw = range / target;
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double f = raw / mag;
    const double nf = f < 1.5 ? 1.0 : f < 3.0 ? 2.0 : f < 7.0 ? 5.0 : 10.0;
    return nf * mag;
}

} // namespace manifold::Rendering
