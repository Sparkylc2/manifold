#pragma once

#include "../constraint.h"

namespace manifold::Solver {
class NoSlipConstraint : public Constraint {
  public:
    NoSlipConstraint();
    ~NoSlipConstraint() override = default;

    void calculate(Output *output, SystemState *state) override;

    void set_body(RigidBody *body);
    void set_radius(double r);
    void set_ks(double ks);
    void set_kd(double kd);

    void reset_offset();

  private:
    double m_radius;
    double m_ks;
    double m_kd;
    double m_offset;
    bool m_offset_initialized;
};

} // namespace manifold::Solver
