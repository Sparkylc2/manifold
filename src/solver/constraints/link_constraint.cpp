#include <manifold/solver/constraints/link_constraint.h>

namespace manifold::Solver {

LinkConstraint::LinkConstraint() : Constraint(2, 2) {
    m_local_1.setZero();
    m_local_2.setZero();

    m_ks = 10.0;
    m_kd = 1.0;

    m_max_force = DBL_MAX;
}

void LinkConstraint::calculate(Output *output, SystemState *state) {
    const int b0 = m_bodies[0]->index;
    const int b1 = m_bodies[1]->index;

    Matrix2d R0 = Rotation2Dd(state->theta[b0]).toRotationMatrix();
    Matrix2d R1 = Rotation2Dd(state->theta[b1]).toRotationMatrix();

    const double w0 = state->v_theta[b0];
    const double w1 = state->v_theta[b1];

    // constraint: anchor0 - anchor1 = 0
    Vector2d C =
        (state->p[b0] + R0 * m_local_1) - (state->p[b1] + R1 * m_local_2);

    // dC/dtheta per body
    Vector2d dC_dt0 = R0 * Vector2d(-m_local_1.y(), m_local_1.x());
    Vector2d dC_dt1 = R1 * Vector2d(-m_local_2.y(), m_local_2.x());

    // J_dot theta columns
    Vector2d Jdot_t0 = -w0 * R0 * m_local_1;
    Vector2d Jdot_t1 = w1 * R1 * m_local_2;

    output->C[0] = C.x();
    output->C[1] = C.y();

    // body 0: +I, +dC_dt0
    output->J[0][0] = 1;
    output->J[0][1] = 0;
    output->J[0][2] = dC_dt0.x();
    output->J[1][0] = 0;
    output->J[1][1] = 1;
    output->J[1][2] = dC_dt0.y();

    // body 1: -I, -dC_dt1
    output->J[0][3] = -1;
    output->J[0][4] = 0;
    output->J[0][5] = -dC_dt1.x();
    output->J[1][3] = 0;
    output->J[1][4] = -1;
    output->J[1][5] = -dC_dt1.y();

    output->J_dot[0][0] = 0;
    output->J_dot[0][1] = 0;
    output->J_dot[0][2] = Jdot_t0.x();
    output->J_dot[1][0] = 0;
    output->J_dot[1][1] = 0;
    output->J_dot[1][2] = Jdot_t0.y();
    output->J_dot[0][3] = 0;
    output->J_dot[0][4] = 0;
    output->J_dot[0][5] = Jdot_t1.x();
    output->J_dot[1][3] = 0;
    output->J_dot[1][4] = 0;
    output->J_dot[1][5] = Jdot_t1.y();

    output->ks[0] = output->ks[1] = m_ks;
    output->kd[0] = output->kd[1] = m_kd;
    output->v_bias[0] = output->v_bias[1] = 0;
    output->limits[0][0] = output->limits[1][0] = -m_max_force;
    output->limits[0][1] = output->limits[1][1] = m_max_force;
}
void LinkConstraint::set_bodies(RigidBody *b1, RigidBody *b2) {
    m_bodies[0] = b1;
    m_bodies[1] = b2;
}

void LinkConstraint::set_body_1(RigidBody *b) { m_bodies[0] = b; }
void LinkConstraint::set_body_2(RigidBody *b) { m_bodies[1] = b; }

void LinkConstraint::set_local_pos1(const Vector2d &p) { m_local_1 = p; }
void LinkConstraint::set_local_pos2(const Vector2d &p) { m_local_2 = p; }

void LinkConstraint::set_ks(double ks) { m_ks = ks; }
void LinkConstraint::set_kd(double kd) { m_kd = kd; }

const Vector2d &LinkConstraint::local_pos1() const { return m_local_1; }
const Vector2d &LinkConstraint::local_pos2() const { return m_local_2; }

} // namespace manifold::Solver
