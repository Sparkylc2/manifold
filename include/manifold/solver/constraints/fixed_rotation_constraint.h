#pragma once

#include "../constraint.h"

namespace manifold::Solver {

class FixedRotationConstraint : public Constraint {
  public:
    FixedRotationConstraint();
    ~FixedRotationConstraint() override = default;

    void calculate(Output *output, SystemState *state) override;

    void set_body(RigidBody *body);

    void set_angle(double angle);
    void set_ks(double ks);
    void set_kd(double kd);

  private:
    double m_angle;
    double m_ks;
    double m_kd;
};

} // namespace manifold::Solver
