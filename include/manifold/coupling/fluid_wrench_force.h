#pragma once

#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Coupling {
using namespace Eigen;

// injects a fluid load into the rigid-body system through the force path
//
// wrench is cached and held constant. not re-running the fluid sim per stage,
// so the fluid load is computed once per coupled tick, and applied unchanged
// across every solver stage
class FluidWrenchForce : public Solver::ForceGenerator {
  public:
    explicit FluidWrenchForce(const Solver::RigidBody *body) : m_body(body) {}

    void apply(Solver::SystemState *state) override {
        const int i = m_body->index;
        if (i < 0)
            return;
        state->f[i] += m_force;
        state->apply_force(m_force, i);
        state->apply_torque(m_torque, i);
    }

    // called once per coupled tick by the driver, after the fluid step
    void set_wrench(const Vector2d &force, double torque) {
        m_force = force;
        m_torque = torque;
    }

  private:
    const Solver::RigidBody *m_body;
    Vector2d m_force = Vector2d::Zero();
    double m_torque = 0.0;
};

} // namespace manifold::Coupling
