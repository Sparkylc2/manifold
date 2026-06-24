#pragma once

#include <manifold/fluid/solid_boundary.h>
#include <manifold/solver/rigid_body.h>

#include <cmath>
#include <functional>
#include <utility>

namespace manifold::Coupling {
using namespace Eigen;

//  presents a Solver::RigidBody to the fluid as a SolidBoundary.
//
//
// shape is supplied as a signed-distance function in the body's local FOR
// (origin at COM, matching RigidBody).
// e.g. a circle of radius r: [r](const Vector2d &l){ return l.norm() - r; }
//
//
// note: the local SDF is evaluated per grid cell per step. std::function may be
// slow so swapping to a templated shape later may be useful
class RigidBodyBoundary : public Fluid::SolidBoundary {
  public:
    using LocalSdf = std::function<double(const Vector2d &)>;

    RigidBodyBoundary(const Solver::RigidBody *body, LocalSdf local_sdf)
        : m_body(body), m_local_sdf(std::move(local_sdf)) {}

    // swap the local shape at runtime (e.g. to change aerofoil profile)
    void set_local_sdf(LocalSdf local_sdf) {
        m_local_sdf = std::move(local_sdf);
    }

    double signed_distance(const Vector2d &x) const override {
        // world -> local, rotating by -theta about COM
        const Vector2d r = x - m_body->p;
        const double c = std::cos(m_body->theta), s = std::sin(m_body->theta);
        const Vector2d l(c * r.x() + s * r.y(), -s * r.x() + c * r.y());
        return m_local_sdf(l);
    }

    void velocity_at(const Vector2d &x, Vector2d *v) const override {
        // v = v_com + omega x r,
        const Vector2d r = x - m_body->p;
        const double w = m_body->v_theta;
        *v = m_body->v + Vector2d(-w * r.y(), w * r.x());
    }

  private:
    const Solver::RigidBody *m_body;
    LocalSdf m_local_sdf;
};

} // namespace manifold::Coupling
