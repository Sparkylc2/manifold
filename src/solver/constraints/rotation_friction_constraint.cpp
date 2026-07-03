#include <manifold/solver/constraints/rotation_friction_constraint.h>

namespace manifold::Solver {

// requires limit-aware solver
RotationFrictionConstraint::RotationFrictionConstraint() : Constraint(1, 1) {
    m_ks = 10.0;
    m_kd = 1.0;
    m_max_torque = DBL_MAX;
    m_min_torque = -DBL_MAX;
}

void RotationFrictionConstraint::calculate(Output *output, SystemState *state) {
    output->C[0] = 0.0;

    // only the angular DOF
    output->J[0][0] = 0.0;
    output->J[0][1] = 0.0;
    output->J[0][2] = 1.0;

    // constant Jacobian
    output->J_dot[0][0] = 0.0;
    output->J_dot[0][1] = 0.0;
    output->J_dot[0][2] = 0.0;

    output->ks[0] = m_ks;
    output->kd[0] = m_kd;
    output->v_bias[0] = 0.0;

    output->limits[0][0] = m_min_torque;
    output->limits[0][1] = m_max_torque;
}

void RotationFrictionConstraint::set_body(RigidBody *body) {
    m_bodies[0] = body;
}

void RotationFrictionConstraint::set_torque_limits(double min_t, double max_t) {
    m_min_torque = min_t;
    m_max_torque = max_t;
}

void RotationFrictionConstraint::set_max_torque(double t) {
    m_max_torque = t;
    m_min_torque = -t;
}

void RotationFrictionConstraint::set_ks(double ks) { m_ks = ks; }
void RotationFrictionConstraint::set_kd(double kd) { m_kd = kd; }

} // namespace manifold::Solver
