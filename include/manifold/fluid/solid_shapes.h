#pragma once

#include <manifold/fluid/solid_boundary.h>

#include <cmath>
#include <functional>
#include <utility>

namespace manifold::Fluid {
using namespace Eigen;

// local-frame signed-distance functions (origin-centered). Compose with a
// transform (StaticBoundary, or Coupling::RigidBodyBoundary for a live body)
// to place/rotate them. Distances are true Euclidean, as the penalization mask
// expects.
using LocalSdf = std::function<double(const Vector2d &)>;

inline LocalSdf circle_sdf(double radius) {
    return [radius](const Vector2d &p) { return p.norm() - radius; };
}

// axis-aligned box, half-extents (hx, hy). hy << hx gives a flat plate;
// hx == hy a square. Exact SDF (negative inside).
inline LocalSdf box_sdf(double hx, double hy) {
    return [hx, hy](const Vector2d &p) {
        const Vector2d d = p.cwiseAbs() - Vector2d(hx, hy);
        const double outside = d.cwiseMax(0.0).norm();
        const double inside = std::min(std::max(d.x(), d.y()), 0.0);
        return outside + inside;
    };
}

// a fixed (non-moving) solid placed at a world pose. velocity is zero, so the
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
        // world -> local (rotate by -theta about pos)
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
