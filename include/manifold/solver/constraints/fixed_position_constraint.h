#pragma once

#include <manifold/solver/constraint.h>

namespace manifold::Solver {

class FixedPositionConstraint : public Constraint {
  public:
    FixedPositionConstraint();
    ~FixedPositionConstraint() override = default;

    void calculate(Output *output, SystemState *state) override;

    void set_body(RigidBody *body);

    void set_world_position(const Vector2d &w);
    void set_local_position(const Vector2d &l);

    void set_ks(double ks);
    void set_kd(double kd);

    const Vector2d &world_position() const;
    const Vector2d &local_position() const;

  private:
    Vector2d m_local;
    Vector2d m_world;

    double m_ks;
    double m_kd;
};
} // namespace manifold::Solver
