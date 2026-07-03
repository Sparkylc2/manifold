#include <manifold/solver/forces/mouse_spring.h>
namespace manifold::Solver {

void MouseSpringForceGenerator::apply(Solver::SystemState *state) {
    if (!m_active || !m_body || m_body->index < 0)
        return;

    const int idx = m_body->index;
    Vector2d attach, v;
    state->local_to_world(m_local, &attach, idx);
    state->velocity_at_point(m_local, &v, idx);

    Vector2d force = m_ks * (m_target - attach) - m_kd * v;
    state->apply_force(m_local, force, idx); // off-center -> force + torque
}

void MouseSpringForceGenerator::set_active(bool a) { m_active = a; }
void MouseSpringForceGenerator::set_body(Solver::RigidBody *b) { m_body = b; }
void MouseSpringForceGenerator::set_target(const Vector2d &t) { m_target = t; }
void MouseSpringForceGenerator::set_local(const Vector2d &l) {
    m_local = l;
} // attach point, body frame

void MouseSpringForceGenerator::set_ks(double ks) { m_ks = ks; }
void MouseSpringForceGenerator::set_kd(double kd) { m_kd = kd; }

bool MouseSpringForceGenerator::active() const { return m_active; }
Solver::RigidBody *MouseSpringForceGenerator::body() const { return m_body; }
const Vector2d &MouseSpringForceGenerator::local() const { return m_local; }
const Vector2d &MouseSpringForceGenerator::target() const { return m_target; }

} // namespace manifold::Solver
