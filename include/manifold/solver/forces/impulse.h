#pragma once

#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

// one-shot force/torque that applies for a single frame then auto-clears.
// use for user kicks, button presses, click impulses, etc.

class ImpulseForceGenerator : public ForceGenerator {
  public:
    void apply(SystemState *state) override {
        if (!m_active || !m_body || m_body->index < 0)
            return;

        const int idx = m_body->index;
        state->f[idx] += m_force;
        state->t[idx] += m_torque;
    }

    void arm_force(const Vector2d &force) {
        m_force = force;
        m_active = true;
    }

    void arm_torque(double torque) {
        m_torque = torque;
        m_active = true;
    }

    void arm(const Vector2d &force, double torque) {
        m_force = force;
        m_torque = torque;
        m_active = true;
    }

    void disarm() {
        m_active = false;
        m_force.setZero();
        m_torque = 0.0;
    }

    void set_body(RigidBody *body) { m_body = body; }
    bool is_active() const { return m_active; }

  private:
    bool m_active = false;
    RigidBody *m_body = nullptr;
    Vector2d m_force = Vector2d::Zero();
    double m_torque = 0.0;
};

} // namespace manifold::Solver
