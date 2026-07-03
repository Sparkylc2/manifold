#pragma once

#include "../constraint.h"
#include <cmath>

namespace manifold::Solver {

class RollingConstraint : public Constraint {
  public:
    RollingConstraint();
    ~RollingConstraint() override = default;

    void calculate(Output *output, SystemState *state) override;

    void set_base_body(RigidBody *body);
    void set_rolling_body(RigidBody *body);

    void set_local_origin(const Vector2d &origin);
    void set_direction(const Vector2d &dir);
    void set_radius(double r);

    void set_ks(double ks);
    void set_kd(double kd);

  private:
    Vector2d m_local;
    Vector2d m_direction;
    double m_radius;
    double m_ks;
    double m_kd;
};

} // namespace manifold::Solver
