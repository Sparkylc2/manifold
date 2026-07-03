#pragma once

#include "../constraint.h"
#include <cfloat>

namespace manifold::Solver {

class GearConstraint : public Constraint {
  public:
    GearConstraint();

    ~GearConstraint() override = default;

    void calculate(Output *output, SystemState *state) override;

    void set_bodies(RigidBody *b0, RigidBody *b1);

    void set_ratio(double ratio);
    double ratio() const;

    void set_neutral(bool n);
    bool neutral() const;

    void set_correct_position(bool c);
    void set_ks(double ks);
    void set_kd(double kd);

    void reset_offset();

  private:
    double m_ratio;
    double m_ks;
    double m_kd;
    bool m_neutral;
    bool m_correct_position;
    double m_offset;
    bool m_offset_initialized;
};

} // namespace manifold::Solver
