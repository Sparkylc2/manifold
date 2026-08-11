#include <manifold/solver/rigid_body_system.h>

#include <assert.h>
#include <cmath>

namespace manifold::Solver {

RigidBodySystem::RigidBodySystem() = default;

RigidBodySystem::~RigidBodySystem() { m_state.clear(); }

void RigidBodySystem::reset() {
    m_bodies.clear();
    m_constraints.clear();
    m_force_generators.clear();
}

void RigidBodySystem::process(double dt, int steps) { /* void */ }

void RigidBodySystem::add_body(RigidBody *body) {
    m_bodies.push_back(body);
    body->index = (int)m_bodies.size() - 1;
}

void RigidBodySystem::remove_body(RigidBody *body) {
    m_bodies[body->index] = m_bodies.back();
    m_bodies[body->index]->index = body->index;
    m_bodies.resize(m_bodies.size() - 1);
}

RigidBody *RigidBodySystem::get_body(int i) {
    assert(i < (int)m_bodies.size());
    return m_bodies[i];
}

void RigidBodySystem::add_constraint(Constraint *constraint) {
    m_constraints.push_back(constraint);
    constraint->m_index = (int)m_constraints.size() - 1;
}

void RigidBodySystem::remove_constraint(Constraint *constraint) {
    m_constraints[constraint->m_index] = m_constraints.back();
    m_constraints[constraint->m_index]->m_index = constraint->m_index;
    m_constraints.resize(m_constraints.size() - 1);
}

void RigidBodySystem::add_force_generator(ForceGenerator *fg) {
    m_force_generators.push_back(fg);
    fg->m_index = (int)m_force_generators.size() - 1;
}

void RigidBodySystem::remove_force_generator(ForceGenerator *fg) {
    m_force_generators[fg->m_index] = m_force_generators.back();
    m_force_generators[fg->m_index]->m_index = fg->m_index;
    m_force_generators.resize(m_force_generators.size() - 1);
}

int RigidBodySystem::get_full_constraint_count() const {
    int count = 0;
    for (Constraint *constraint : m_constraints) {
        count += constraint->constraint_count();
    }
    return count;
}

void RigidBodySystem::populate_state() {
    const int num_b = get_body_count();
    const int num_c_eq = get_full_constraint_count();

    m_state.resize(num_b, num_c_eq);

    for (int i = 0; i < num_b; i++) {
        m_state.a[i].setZero();

        m_state.v[i] = m_bodies[i]->v;
        m_state.p[i] = m_bodies[i]->p;

        m_state.a_theta[i] = 0;
        m_state.v_theta[i] = m_bodies[i]->v_theta;
        m_state.theta[i] = m_bodies[i]->theta;

        m_state.m[i] = m_bodies[i]->m;
    }
}

void RigidBodySystem::populate_mass_matrices(MatrixXd *M, MatrixXd *M_inv) {
    const int num_b = get_body_count();

    M->resize(1, 3 * num_b);
    M_inv->resize(1, 3 * num_b);

    for (int i = 0; i < num_b; i++) {
        M->block<1, 3>(0, i * 3) << m_bodies[i]->m, m_bodies[i]->m,
            m_bodies[i]->I;
    }

    *M_inv = M->cwiseInverse();
}

void RigidBodySystem::process_forces() {
    const int num_f = get_force_generator_count();
    const int num_b = get_body_count();

    for (int i = 0; i < num_b; i++) {
        m_state.f[i].setZero();
        m_state.t[i] = 0.0;
    }

    for (int i = 0; i < num_f; i++) {
        m_force_generators[i]->apply(&m_state);
    }
}
} // namespace manifold::Solver
