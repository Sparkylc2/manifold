#pragma once

#include <algorithm>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class ControlTorque : public ForceGenerator {
  public:
    void apply(SystemState *state) override {
        double tau = std::clamp(m_torque, -m_max_torque, m_max_torque);

        if (m_body_0 && m_body_0->index >= 0)
            state->t[m_body_0->index] += tau;
        if (m_body_1 && m_body_1->index >= 0)
            state->t[m_body_1->index] -= tau;
    }

    void set_bodies(RigidBody *b0, RigidBody *b1) {
        m_body_0 = b0;
        m_body_1 = b1;
    }

    void set_torque(double tau) { m_torque = tau; }
    double torque() const { return m_torque; }

    void set_max_torque(double t) { m_max_torque = t; }
    double max_torque() const { return m_max_torque; }

  private:
    RigidBody *m_body_0 = nullptr;
    RigidBody *m_body_1 = nullptr;
    double m_torque = 0.0;
    double m_max_torque = 50.0;
};

} // namespace manifold::Solver
