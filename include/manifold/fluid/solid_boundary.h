#pragma once

#include <Eigen/Dense>

namespace manifold::Fluid {
using namespace Eigen;

// view of a solid embedded in the fluid
// coordinates are world-space, distances/velocities are in physical units
class SolidBoundary {
  public:
    virtual ~SolidBoundary() = default;

    // signed distance to the solid surface at world point x
    // negative inside the solid, positive outside, ~0 on the surface
    virtual double signed_distance(const Vector2d &x) const = 0;

    // world-space velocity of the solid material at world point x
    // (ie. v_com + omega x r)
    virtual void velocity_at(const Vector2d &x, Vector2d *v) const = 0;

    // outward surface normal at world point x (unit)
    // defaults to the gradient of the SDF by central differences
    virtual void normal_at(const Vector2d &x, Vector2d *n) const {
        const double h = 1e-4;
        const double dx = signed_distance(x + Vector2d(h, 0)) -
                          signed_distance(x - Vector2d(h, 0));
        const double dy = signed_distance(x + Vector2d(0, h)) -
                          signed_distance(x - Vector2d(0, h));
        Vector2d g(dx, dy);
        const double len = g.norm();
        *n = (len > 1e-12) ? (g / len).eval() : Vector2d(1.0, 0.0);
    }

    // inside test derived from the sign of the SDF
    bool inside(const Vector2d &x) const { return signed_distance(x) < 0.0; }
};

} // namespace manifold::Fluid
