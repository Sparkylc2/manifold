#pragma once

#include "../constraint.h"
#include <cfloat>
#include <cmath>

namespace manifold::Solver {

class LinkConstraint : public Constraint {
  public:
    LinkConstraint();
    ~LinkConstraint() override = default;

    void calculate(Output *output, SystemState *state) override;
    void set_bodies(RigidBody *b1, RigidBody *b2);

    void set_body_1(RigidBody *b);
    void set_body_2(RigidBody *b);

    void set_local_pos1(const Vector2d &p);
    void set_local_pos2(const Vector2d &p);

    void set_ks(double ks);
    void set_kd(double kd);

    const Vector2d &local_pos1() const;
    const Vector2d &local_pos2() const;

  private:
    double m_max_force;

    Vector2d m_local_1;
    Vector2d m_local_2;

    double m_ks;
    double m_kd;
};

} // namespace manifold::Solver
