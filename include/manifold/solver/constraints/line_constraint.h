#pragma once

#include "../constraint.h"
#include <cmath>

namespace manifold::Solver {

class LineConstraint : public Constraint {
  public:
    LineConstraint() : Constraint(1, 1) {
        m_local.setZero();
        m_p0.setZero();
        m_d.setZero();

        m_ks = 10.0;
        m_kd = 1.0;
    }

    ~LineConstraint() override = default;

    void calculate(Output *output, SystemState *state) override {
        const int b = m_bodies[0]->index;
        const double q3_dot = state->v_theta[b];

        Matrix2d R = Rotation2Dd(state->theta[b]).toRotationMatrix();
        Vector2d perp(-m_d.y(), m_d.x());

        Vector2d world_pos = state->p[b] + R * m_local;
        double C = (world_pos - m_p0).dot(perp);

        double dC_dtheta = (R * Vector2d(-m_local.y(), m_local.x())).dot(perp);
        double Jdot_theta = (-q3_dot * R * m_local).dot(perp);

        output->C[0] = C;

        output->J[0][0] = perp.x();
        output->J[0][1] = perp.y();
        output->J[0][2] = dC_dtheta;

        output->J_dot[0][0] = 0;
        output->J_dot[0][1] = 0;
        output->J_dot[0][2] = Jdot_theta;

        output->ks[0] = m_ks;
        output->kd[0] = m_kd;
        output->v_bias[0] = 0;
        no_limits(output);
    }

    void set_body(RigidBody *body) { m_bodies[0] = body; }

    void set_line(Vector2d p0, Vector2d d) {
        m_p0 = p0;
        m_d = d;
    }
    void set_local_pos(Vector2d l) { m_local = l; }

    void set_kd(double kd) { m_kd = kd; }
    void set_ks(double ks) { m_ks = ks; }

    const Vector2d &line_origin() const { return m_p0; }
    const Vector2d &line_direction() const { return m_d; }

  private:
    Vector2d m_local;
    Vector2d m_p0;
    Vector2d m_d;

    double m_ks;
    double m_kd;
};

} // namespace manifold::Solver
