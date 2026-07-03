
#include <manifold/solver/forces/constant_speed_motor.h>

namespace manifold::Solver {

void ConstantSpeedMotor::apply(SystemState *state) {
    if (!m_body_1 || m_body_1->index < 0)
        return;

    double v0 = 0.0, a0 = 0.0;
    if (m_body_0 && m_body_0->index >= 0) {
        v0 = state->v_theta[m_body_0->index];
        a0 = state->a_theta[m_body_0->index];
    }

    const double rel_v = state->v_theta[m_body_1->index] - v0;
    const double rel_a = state->a_theta[m_body_1->index] - a0;

    const double error = m_speed - rel_v;
    const double torque =
        std::clamp(m_ks * error - m_kd * rel_a, -m_max_torque, m_max_torque);

    if (m_body_0 && m_body_0->index >= 0)
        state->t[m_body_0->index] -= torque;
    state->t[m_body_1->index] += torque;
} // namespace manifold::Solver

void ConstantSpeedMotor::set_bodies(RigidBody *b0, RigidBody *b1) {
    m_body_0 = b0;
    m_body_1 = b1;
}

void ConstantSpeedMotor::set_speed(double speed) { m_speed = speed; }
double ConstantSpeedMotor::speed() const { return m_speed; }

void ConstantSpeedMotor::set_max_torque(double t) { m_max_torque = t; }
void ConstantSpeedMotor::set_ks(double ks) { m_ks = ks; }
void ConstantSpeedMotor::set_kd(double kd) { m_kd = kd; }

} // namespace manifold::Solver
