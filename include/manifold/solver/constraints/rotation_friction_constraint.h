#pragma once

#include "../constraint.h"
#include <cfloat>

namespace manifold::Solver {

// requires limit-aware solver

class RotationFrictionConstraint : public Constraint {
  public:
    RotationFrictionConstraint();
    ~RotationFrictionConstraint() override = default;

    void calculate(Output *output, SystemState *state) override;

    void set_body(RigidBody *body);

    void set_torque_limits(double min_t, double max_t);

    void set_max_torque(double t);

    void set_ks(double ks);
    void set_kd(double kd);

  private:
    double m_ks;
    double m_kd;
    double m_max_torque;
    double m_min_torque;
};

} // namespace manifold::Solver
