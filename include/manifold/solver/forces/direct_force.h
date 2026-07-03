#pragma once

#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Solver {

class DirectForceGenerator : public ForceGenerator {
  public:
    DirectForceGenerator();
    ~DirectForceGenerator() override = default;

    void apply(SystemState *state) override;

    void set_force(const Vector2d &f);
    void set_local_position(const Vector2d &l);
    void set_body(RigidBody *body);

    Vector2d get_force() const;

  private:
    RigidBody *m_body;
    Vector2d m_f; // force
    Vector2d m_l; // local pos
};

} // namespace manifold::Solver
