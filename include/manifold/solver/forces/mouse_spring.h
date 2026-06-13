#pragma once
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class MouseSpringForceGenerator : public ForceGenerator {
  public:
    void apply(Solver::SystemState *state) override {
        if (!m_active || !m_body || m_body->index < 0)
            return;

        Vector2d r = m_target - state->p[m_body->index];
        Vector2d v = state->v[m_body->index];

        Vector2d force = m_ks * r - m_kd * v;
        state->f[m_body->index] += force;
    }

    void set_active(bool a) { m_active = a; }
    void set_body(Solver::RigidBody *b) { m_body = b; }
    void set_target(const Vector2d &t) { m_target = t; }

    void set_ks(double ks) { m_ks = ks; }
    void set_kd(double kd) { m_kd = kd; }

  private:
    bool m_active = false;
    Solver::RigidBody *m_body = nullptr;
    Vector2d m_target = Vector2d::Zero();
    double m_ks = 100.0;
    double m_kd = 0.0;
};
} // namespace manifold::Solver
