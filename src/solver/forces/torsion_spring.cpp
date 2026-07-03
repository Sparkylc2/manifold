#include <manifold/solver/forces/torsion_spring.h>
namespace manifold::Solver {

void TorsionSpring::apply(SystemState *state) {
    if (!m_body || m_body->index < 0)
        return;
    const int i = m_body->index;
    const double theta = state->theta[i];
    const double omega = state->v_theta[i];
    const double tau = -m_ks * (theta - m_rest_angle) - m_kd * omega;
    state->apply_torque(tau, i);
}

double TorsionSpring::energy() const {
    if (!m_body)
        return 0.0;
    const double d = m_body->theta - m_rest_angle;
    return 0.5 * m_ks * d * d;
}

void TorsionSpring::set_body(RigidBody *b) { m_body = b; }
void TorsionSpring::set_rest_angle(double a) { m_rest_angle = a; }
void TorsionSpring::set_ks(double ks) { m_ks = ks; }
void TorsionSpring::set_kd(double kd) { m_kd = kd; }

} // namespace manifold::Solver
