#include <manifold/solver/forces/impulse.h>

namespace manifold::Solver {

void ImpulseForceGenerator::apply(SystemState *state) {
    if (!m_active || !m_body || m_body->index < 0)
        return;

    const int idx = m_body->index;
    state->f[idx] += m_force;
    state->t[idx] += m_torque;
}

void ImpulseForceGenerator::arm_force(const Vector2d &force) {
    m_force = force;
    m_active = true;
}

void ImpulseForceGenerator::arm_torque(double torque) {
    m_torque = torque;
    m_active = true;
}

void ImpulseForceGenerator::arm(const Vector2d &force, double torque) {
    m_force = force;
    m_torque = torque;
    m_active = true;
}

void ImpulseForceGenerator::disarm() {
    m_active = false;
    m_force.setZero();
    m_torque = 0.0;
}

void ImpulseForceGenerator::set_body(RigidBody *body) { m_body = body; }
bool ImpulseForceGenerator::is_active() const { return m_active; }

} // namespace manifold::Solver
