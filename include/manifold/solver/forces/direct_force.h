#pragma once

#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class DirectForceGenerator : public ForceGenerator {
  public:
    DirectForceGenerator() {
        m_f.setZero();
        m_l.setZero();

        m_body = nullptr;
    }
    ~DirectForceGenerator() override = default;

    void apply(SystemState *state) override {
        state->apply_force(m_l, m_f, m_body->index);
    }

    void set_force(const Vector2d &f) { m_f = f; }
    void set_local_position(const Vector2d &l) { m_l = l; }
    void set_body(RigidBody *body) { m_body = body; }

    Vector2d get_force() const { return m_f; }

  private:
    RigidBody *m_body;
    Vector2d m_f; // force
    Vector2d m_l; // local pos
};

} // namespace manifold::Solver
