#include <manifold/solver/forces/control_torque.h>

namespace manifold::Solver {
void ControlTorque::apply(SystemState *state) {
    double tau = std::clamp(m_torque, -m_max_torque, m_max_torque);

    if (m_body_0 && m_body_0->index >= 0)
        state->t[m_body_0->index] += tau;
    if (m_body_1 && m_body_1->index >= 0)
        state->t[m_body_1->index] -= tau;
}

void ControlTorque::set_bodies(RigidBody *b0, RigidBody *b1) {
    m_body_0 = b0;
    m_body_1 = b1;
}

void ControlTorque::set_torque(double tau) { m_torque = tau; }
double ControlTorque::torque() const { return m_torque; }

void ControlTorque::set_max_torque(double t) { m_max_torque = t; }
double ControlTorque::max_torque() const { return m_max_torque; }

} // namespace manifold::Solver
