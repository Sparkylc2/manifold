#include <manifold/solver/constraints/no_slip_constraint.h>

namespace manifold::Solver {
NoSlipConstraint::NoSlipConstraint() : Constraint(1, 1) {
    m_radius = 1.0;
    m_ks = 0.0;
    m_kd = 1.0;
    m_offset = 0.0;
    m_offset_initialized = false;
}

void NoSlipConstraint::calculate(Output *output, SystemState *state) {
    const int b = m_bodies[0]->index;

    if (!m_offset_initialized) {
        m_offset = state->p[b].x() + m_radius * state->theta[b];
        m_offset_initialized = true;
    }

    output->C[0] = state->p[b].x() + m_radius * state->theta[b] - m_offset;

    // J = [1, 0, radius]
    output->J[0][0] = 1.0;
    output->J[0][1] = 0.0;
    output->J[0][2] = m_radius;

    output->J_dot[0][0] = 0.0;
    output->J_dot[0][1] = 0.0;
    output->J_dot[0][2] = 0.0;

    output->ks[0] = m_ks;
    output->kd[0] = m_kd;
    output->v_bias[0] = 0.0;

    no_limits(output);
}

void NoSlipConstraint::set_body(RigidBody *body) { m_bodies[0] = body; }
void NoSlipConstraint::set_radius(double r) { m_radius = r; }
void NoSlipConstraint::set_ks(double ks) { m_ks = ks; }
void NoSlipConstraint::set_kd(double kd) { m_kd = kd; }

void NoSlipConstraint::reset_offset() { m_offset_initialized = false; }

} // namespace manifold::Solver
