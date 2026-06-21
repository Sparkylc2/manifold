#pragma once
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class MouseSpringForceGenerator : public ForceGenerator {
  public:
    void apply(Solver::SystemState *state) override {
        if (!m_active || !m_body || m_body->index < 0)
            return;

        const int idx = m_body->index;
        Vector2d attach, v;
        state->local_to_world(m_local, &attach, idx);
        state->velocity_at_point(m_local, &v, idx);

        Vector2d force = m_ks * (m_target - attach) - m_kd * v;
        state->apply_force(m_local, force, idx); // off-center -> force + torque
    }

    void set_active(bool a) { m_active = a; }
    void set_body(Solver::RigidBody *b) { m_body = b; }
    void set_target(const Vector2d &t) { m_target = t; }
    void set_local(const Vector2d &l) { m_local = l; } // attach point, body frame

    void set_ks(double ks) { m_ks = ks; }
    void set_kd(double kd) { m_kd = kd; }

    bool active() const { return m_active; }
    Solver::RigidBody *body() const { return m_body; }
    const Vector2d &local() const { return m_local; }
    const Vector2d &target() const { return m_target; }

  private:
    bool m_active = false;
    Solver::RigidBody *m_body = nullptr;
    Vector2d m_target = Vector2d::Zero();
    Vector2d m_local = Vector2d::Zero();
    double m_ks = 100.0;
    double m_kd = 0.0;
};
} // namespace manifold::Solver
