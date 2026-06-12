#pragma once

#include <manifold/solver/constraint.h>

namespace manifold::Solver {

class FixedPositionConstraint : public Constraint {
  public:
    FixedPositionConstraint() : Constraint(2, 1) {
        m_local.setZero();
        m_world.setZero();

        m_ks = 10.0;
        m_kd = 1.0;
    }
    ~FixedPositionConstraint() override = default;

    void calculate(Output *output, SystemState *state) override {

        const int b = m_bodies[0]->index;
        const double q3_dot = state->v_theta[b];

        Matrix2d R = Rotation2Dd(state->theta[b]).toRotationMatrix();

        Vector2d C = state->p[b] + R * m_local - m_world;
        Vector2d dC_dtheta = R * Vector2d(-m_local.y(), m_local.x());
        Vector2d Jdot_theta = -q3_dot * R * m_local;

        output->C[0] = C.x();
        output->C[1] = C.y();

        output->J[0][0] = 1;
        output->J[0][1] = 0;
        output->J[0][2] = dC_dtheta.x();
        output->J[1][0] = 0;
        output->J[1][1] = 1;
        output->J[1][2] = dC_dtheta.y();

        output->J_dot[0][0] = 0;
        output->J_dot[0][1] = 0;
        output->J_dot[0][2] = Jdot_theta.x();
        output->J_dot[1][0] = 0;
        output->J_dot[1][1] = 0;
        output->J_dot[1][2] = Jdot_theta.y();

        output->ks[0] = output->ks[1] = m_ks;
        output->kd[0] = output->kd[1] = m_kd;
        output->v_bias[0] = output->v_bias[1] = 0;
        no_limits(output);
    };

    void set_body(RigidBody *body) { m_bodies[0] = body; }

    void set_world_position(const Vector2d &w) { m_world = w; }
    void set_local_position(const Vector2d &l) { m_local = l; }

    void set_ks(double ks) { m_ks = ks; }
    void set_kd(double kd) { m_kd = kd; }

    const Vector2d &world_position() const { return m_world; }
    const Vector2d &local_position() const { return m_local; }

  private:
    Vector2d m_local;
    Vector2d m_world;

    double m_ks;
    double m_kd;
};
} // namespace manifold::Solver
