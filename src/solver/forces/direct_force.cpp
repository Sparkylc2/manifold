#include <manifold/solver/forces/direct_force.h>

namespace manifold::Solver {

DirectForceGenerator::DirectForceGenerator() {
    m_f.setZero();
    m_l.setZero();

    m_body = nullptr;
}

void DirectForceGenerator::apply(SystemState *state) {
    state->apply_force(m_l, m_f, m_body->index);
}

void DirectForceGenerator::set_force(const Vector2d &f) { m_f = f; }
void DirectForceGenerator::set_local_position(const Vector2d &l) { m_l = l; }
void DirectForceGenerator::set_body(RigidBody *body) { m_body = body; }

Vector2d DirectForceGenerator::get_force() const { return m_f; }

} // namespace manifold::Solver
