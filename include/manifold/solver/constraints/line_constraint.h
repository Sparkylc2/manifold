#pragma once

#include "../constraint.h"
#include <cmath>

namespace manifold::Solver {

class LineConstraint : public Constraint {
  public:
    LineConstraint();

    ~LineConstraint() override = default;

    void calculate(Output *output, SystemState *state) override;

    void set_body(RigidBody *body);

    void set_line(Vector2d p0, Vector2d d);
    void set_local_pos(Vector2d l);

    void set_kd(double kd);
    void set_ks(double ks);

    const Vector2d &line_origin() const;
    const Vector2d &line_direction() const;

  private:
    Vector2d m_local;
    Vector2d m_p0;
    Vector2d m_d;

    double m_ks;
    double m_kd;
};

} // namespace manifold::Solver
