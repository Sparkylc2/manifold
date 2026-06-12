#pragma once

#include "../constraint.h"

namespace manifold::Solver {

class FixedRotationConstraint : public Constraint {
  public:
    FixedRotationConstraint() : Constraint(1, 1) {
        m_angle = 0.0;
        m_ks = 10.0;
        m_kd = 1.0;
    }

    ~FixedRotationConstraint() override = default;

    void calculate(Output *output, SystemState *state) override {
        const int b = m_bodies[0]->index;
        const double q3 = state->theta[b];

        const double C = q3 - m_angle;

        output->C[0] = C;

        output->J[0][0] = 0;
        output->J[0][1] = 0;
        output->J[0][2] = 1.0;

        output->J_dot[0][0] = 0;
        output->J_dot[0][1] = 0;
        output->J_dot[0][2] = 0;

        output->kd[0] = m_kd;
        output->ks[0] = m_ks;

        output->v_bias[0] = 0;

        no_limits(output);
    }

    void set_body(RigidBody *body) { m_bodies[0] = body; }

    void set_angle(double angle) { m_angle = angle; }
    void set_ks(double ks) { m_ks = ks; }
    void set_kd(double kd) { m_kd = kd; }

  private:
    double m_angle;
    double m_ks;
    double m_kd;
};

} // namespace manifold::Solver
