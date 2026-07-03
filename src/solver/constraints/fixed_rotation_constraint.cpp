#include <manifold/solver/constraints/fixed_rotation_constraint.h>

namespace manifold::Solver {

FixedRotationConstraint::FixedRotationConstraint() : Constraint(1, 1) {
    m_angle = 0.0;
    m_ks = 10.0;
    m_kd = 1.0;
}

void FixedRotationConstraint::calculate(Output *output, SystemState *state) {
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

void FixedRotationConstraint::set_body(RigidBody *body) { m_bodies[0] = body; }

void FixedRotationConstraint::set_angle(double angle) { m_angle = angle; }
void FixedRotationConstraint::set_ks(double ks) { m_ks = ks; }
void FixedRotationConstraint::set_kd(double kd) { m_kd = kd; }

} // namespace manifold::Solver
