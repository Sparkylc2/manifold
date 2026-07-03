#pragma once

#include "../constraint.h"
#include <cmath>

namespace manifold::Solver {

class DistanceConstraint : public Constraint {
  public:
    DistanceConstraint();

    ~DistanceConstraint() override = default;

    void calculate(Output *output, SystemState *state) override;

    void set_bodies(RigidBody *a, RigidBody *b);

    void set_distance(double d);

    // automatically compute from current body positions
    void set_distance_from_bodies();

    double distance() const;

    void set_ks(double ks);
    void set_kd(double kd);

  private:
    double m_distance;
    double m_ks;
    double m_kd;
};

} // namespace manifold::Solver
