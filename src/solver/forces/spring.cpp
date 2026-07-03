#include <manifold/solver/forces/spring.h>
namespace manifold::Solver {

void Spring::apply(SystemState *state) {
    if (!m_body_1 || !m_body_2)
        return;

    Vector2d x1, x2, v1 = Vector2d::Zero(), v2 = Vector2d::Zero();

    if (m_body_1->index != -1) {
        state->local_to_world(m_p1, &x1, m_body_1->index);
        state->velocity_at_point(m_p1, &v1, m_body_1->index);
    } else {
        m_body_1->local_to_world(m_p1, &x1);
    }

    if (m_body_2->index != -1) {
        state->local_to_world(m_p2, &x2, m_body_2->index);
        state->velocity_at_point(m_p2, &v2, m_body_2->index);
    } else {
        m_body_2->local_to_world(m_p2, &x2);
    }

    Vector2d delta = x2 - x1;
    double len = delta.norm();

    Vector2d dir = (len >= 1e-2) ? (delta / len).eval() : Vector2d::Zero();

    Vector2d rel_v = v2 - v1;
    double stretch = len - m_rest_length;

    Vector2d f = dir * stretch * m_ks + rel_v * m_kd;

    if (m_body_1->index != -1)
        state->apply_force(m_p1, f, m_body_1->index);
    if (m_body_2->index != -1)
        state->apply_force(m_p2, -f, m_body_2->index);
}

void Spring::get_ends(Vector2d *w1, Vector2d *w2) const {
    if (!m_body_1 || !m_body_2)
        return;
    m_body_1->local_to_world(m_p1, w1);
    m_body_2->local_to_world(m_p2, w2);
}

double Spring::energy() const {
    if (!m_body_1 || !m_body_2)
        return 0;
    Vector2d w1, w2;
    m_body_1->local_to_world(m_p1, &w1);
    m_body_2->local_to_world(m_p2, &w2);
    double stretch = (w2 - w1).norm() - m_rest_length;
    return 0.5 * m_ks * stretch * stretch;
}

void Spring::set_bodies(RigidBody *b1, RigidBody *b2) {
    m_body_1 = b1;
    m_body_2 = b2;
}
void Spring::set_local_pos1(Vector2d p) { m_p1 = p; }
void Spring::set_local_pos2(Vector2d p) { m_p2 = p; }

void Spring::set_kd(double kd) { m_kd = kd; }
void Spring::set_ks(double ks) { m_ks = ks; }

void Spring::set_rest_length(double rest_length) {
    m_rest_length = rest_length;
}

} // namespace manifold::Solver
