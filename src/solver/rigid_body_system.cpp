#include <manifold/solver/rigid_body_system.h>

#include <assert.h>
#include <chrono>
#include <cmath>

namespace manifold::Solver {

RigidBodySystem::RigidBodySystem() {
    m_ode_solve_microseconds = new long long[profiling_samples];
    m_constraint_solve_microseconds = new long long[profiling_samples];
    m_force_eval_microseconds = new long long[profiling_samples];
    m_constraint_eval_microseconds = new long long[profiling_samples];
    m_frame_index = 0;

    for (int i = 0; i < profiling_samples; i++) {
        m_ode_solve_microseconds[i] = -1;
        m_constraint_solve_microseconds[i] = -1;
        m_force_eval_microseconds[i] = -1;
        m_constraint_eval_microseconds[i] = -1;
    }
}

RigidBodySystem::~RigidBodySystem() {
    delete[] m_ode_solve_microseconds;
    delete[] m_constraint_solve_microseconds;
    delete[] m_force_eval_microseconds;
    delete[] m_constraint_eval_microseconds;

    m_state.clear();
}

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
    // neat way to remove bodies i can't lie
    m_bodies[body->index] = m_bodies.back();
    m_bodies[body->index]->index = body->index;
    m_bodies.resize(m_bodies.size() - 1);
}

RigidBody *RigidBodySystem::get_body(int i) {
    assert(i < m_bodies.size());
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

float RigidBodySystem::find_avg(long long *samples) {
    long long accum = 0;
    int count = 0;
    for (int i = 0; i < profiling_samples; i++) {
        if (samples[i] != -1) {
            accum += samples[i];
            count++;
        }
    }
    if (count == 0)
        return 0;
    else
        return (float)accum / count;
}

float RigidBodySystem::get_ode_solve_microseconds() const {
    return find_avg(m_ode_solve_microseconds);
}

float RigidBodySystem::get_constraint_solve_microseconds() const {
    return find_avg(m_constraint_solve_microseconds);
}
float RigidBodySystem::get_constraint_eval_microseconds() const {
    return find_avg(m_constraint_eval_microseconds);
}

float RigidBodySystem::get_force_eval_microseconds() const {
    return find_avg(m_force_eval_microseconds);
}

void RigidBodySystem::populate_state() {
    const int num_b = get_body_count();
    const int num_c_eq = get_full_constraint_count();
    const int num_c = get_constraint_count();

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

    for (int i = 0, curr_c_row = 0; i < num_c; i++) {
        m_state.index_map[i] = curr_c_row;
        curr_c_row += m_constraints[i]->constraint_count();
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
