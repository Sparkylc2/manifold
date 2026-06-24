#pragma once

#include <manifold/fluid/solid_boundary.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

namespace manifold::Fluid {
using namespace Eigen;

// local-frame signed-distance functions (origin-centered).
using LocalSdf = std::function<double(const Vector2d &)>;

inline LocalSdf circle_sdf(double radius) {
    return [radius](const Vector2d &p) { return p.norm() - radius; };
}

// axis-aligned box, half-extents (hx, hy)
inline LocalSdf box_sdf(double hx, double hy) {
    return [hx, hy](const Vector2d &p) {
        const Vector2d d = p.cwiseAbs() - Vector2d(hx, hy);
        const double outside = d.cwiseMax(0.0).norm();
        const double inside = std::min(std::max(d.x(), d.y()), 0.0);
        return outside + inside;
    };
}

// closed outline of a NACA 4-digit aerofoil in the local frame
inline std::vector<Vector2d> naca_points(int code, double chord, int panels) {
    const double m = (code / 1000 % 10) / 100.0; // max camber fraction
    const double p = (code / 100 % 10) / 10.0;   // camber position fraction
    const double th = (code % 100) / 100.0;      // max thickness fraction
    const int n = std::max(panels, 4);

    std::vector<Vector2d> up, lo;
    up.reserve(n + 1);
    lo.reserve(n + 1);
    for (int i = 0; i <= n; i++) {

        const double beta = M_PI * (double)i / (double)n;
        const double x = 0.5 * (1.0 - std::cos(beta));
        // last coefficient 0.1036 (not 0.1015) closes the trailing edge to a
        // sharp point so the outline meets and the fill has no gap there
        const double yt = 5.0 * th *
                          (0.2969 * std::sqrt(x) - 0.1260 * x - 0.3516 * x * x +
                           0.2843 * x * x * x - 0.1036 * x * x * x * x);

        double yc = 0.0, dyc = 0.0;
        if (m > 0.0 && p > 0.0) {
            if (x < p) {
                yc = m / (p * p) * (2.0 * p * x - x * x);
                dyc = 2.0 * m / (p * p) * (p - x);
            } else {
                yc = m / ((1.0 - p) * (1.0 - p)) *
                     ((1.0 - 2.0 * p) + 2.0 * p * x - x * x);
                dyc = 2.0 * m / ((1.0 - p) * (1.0 - p)) * (p - x);
            }
        }
        const double theta = std::atan(dyc);
        const double ct = std::cos(theta), st = std::sin(theta);

        up.push_back(
            Vector2d(((x - yt * st) - 0.5) * chord, (yc + yt * ct) * chord));
        lo.push_back(
            Vector2d(((x + yt * st) - 0.5) * chord, (yc - yt * ct) * chord));
    }

    std::vector<Vector2d> out;
    out.reserve(2 * (n + 1));

    for (auto &q : up) {
        out.push_back(q);
    }

    for (auto it = lo.rbegin(); it != lo.rend(); ++it) {
        out.push_back(*it);
    }

    return out;
}

// distance from p to segment ab
inline double segment_distance(const Vector2d &p, const Vector2d &a,
                               const Vector2d &b) {
    const Vector2d ab = b - a, ap = p - a;
    const double len2 = ab.squaredNorm();
    double t = (len2 > 1e-20) ? ap.dot(ab) / len2 : 0.0;
    t = std::clamp(t, 0.0, 1.0);

    return (p - (a + t * ab)).norm();
}

// even-odd point-in-polygon test
inline bool point_in_polygon(const Vector2d &p,
                             const std::vector<Vector2d> &v) {
    bool in = false;
    for (size_t i = 0, j = v.size() - 1; i < v.size(); j = i++) {
        if (((v[i].y() > p.y()) != (v[j].y() > p.y())) &&
            (p.x() < (v[j].x() - v[i].x()) * (p.y() - v[i].y()) /
                             (v[j].y() - v[i].y()) +
                         v[i].x())) {

            in = !in;
        }
    }
    return in;
}

// exact-ish signed distance to a closed polygon
inline LocalSdf polygon_sdf(std::vector<Vector2d> pts) {
    return [pts = std::move(pts)](const Vector2d &p) {
        double d = 1e30;
        for (size_t i = 0, j = pts.size() - 1; i < pts.size(); j = i++)
            d = std::min(d, segment_distance(p, pts[j], pts[i]));
        return point_in_polygon(p, pts) ? -d : d;
    };
}

// signed area (positive if the points wind counter-clockwise)
inline double polygon_area(const std::vector<Vector2d> &v) {
    double a = 0.0;
    for (size_t i = 0, j = v.size() - 1; i < v.size(); j = i++)
        a += v[j].x() * v[i].y() - v[i].x() * v[j].y();
    return 0.5 * a;
}

// area centroid
inline Vector2d polygon_centroid(const std::vector<Vector2d> &v) {
    double a = 0.0, cx = 0.0, cy = 0.0;

    for (size_t i = 0, j = v.size() - 1; i < v.size(); j = i++) {
        const double cross = v[j].x() * v[i].y() - v[i].x() * v[j].y();
        a += cross;
        cx += (v[j].x() + v[i].x()) * cross;
        cy += (v[j].y() + v[i].y()) * cross;
    }

    if (std::abs(a) < 1e-12) {
        return Vector2d::Zero();
    }

    return Vector2d(cx / (3.0 * a), cy / (3.0 * a));
}

// mass moment of inertia about the centroid, for a uniform-density polygon of
// total mass `mass`
inline double polygon_inertia(const std::vector<Vector2d> &v, double mass) {
    double sa = 0.0, jx = 0.0, jy = 0.0;
    for (size_t i = 0, j = v.size() - 1; i < v.size(); j = i++) {
        const double xj = v[j].x(), yj = v[j].y(), xi = v[i].x(), yi = v[i].y();
        const double cross = xj * yi - xi * yj;
        sa += cross;
        jx += cross * (yj * yj + yj * yi + yi * yi);
        jy += cross * (xj * xj + xj * xi + xi * xi);
    }

    const double area = 0.5 * sa;
    if (std::abs(area) < 1e-12) {
        return 0.0;
    }
    const double Jo = (jx + jy) / 12.0;
    const Vector2d c = polygon_centroid(v);
    const double Jc = Jo - area * (c.x() * c.x() + c.y() * c.y());

    return mass * Jc / area;
}

inline LocalSdf naca_sdf(int code = 12, double chord = 1.0, int panels = 12) {
    return polygon_sdf(naca_points(code, chord, panels));
}

// fixed solid. velocity is zero, so the
// fluid sees no-slip against a stationary wall/plate/square.
class StaticBoundary : public SolidBoundary {
  public:
    StaticBoundary(const Vector2d &pos, double theta, LocalSdf sdf)
        : m_pos(pos), m_c(std::cos(theta)), m_s(std::sin(theta)),
          m_sdf(std::move(sdf)) {}

    void set_pose(const Vector2d &pos, double theta) {
        m_pos = pos;
        m_c = std::cos(theta);
        m_s = std::sin(theta);
    }

    double signed_distance(const Vector2d &x) const override {
        const Vector2d r = x - m_pos;
        const Vector2d l(m_c * r.x() + m_s * r.y(), -m_s * r.x() + m_c * r.y());
        return m_sdf(l);
    }

    void velocity_at(const Vector2d &, Vector2d *v) const override {
        v->setZero();
    }

  private:
    Vector2d m_pos;
    double m_c, m_s; // cached cos/sin(theta)
    LocalSdf m_sdf;
};

} // namespace manifold::Fluid
