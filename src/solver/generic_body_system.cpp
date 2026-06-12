#include <cassert>
#include <manifold/solver/generic_body_system.h>

namespace manifold::Solver {

GenericRigidBodySystem::GenericRigidBodySystem() = default;

void GenericRigidBodySystem::initialize(SLESolver *sle_solver,
                                        ODESolver *ode_solver) {
    m_sle_solver = sle_solver;
    m_ode_solver = ode_solver;
    m_iv.lambda.resize(0);
}

void GenericRigidBodySystem::process(double dt, int steps) {
#ifdef MANIFOLD_PROFILE
    long long ode_time = 0, constraint_time = 0, force_time = 0, eval_time = 0;
#endif

    populate_state();

    // build M_inv once per frame
    const int n = get_body_count();
    m_iv.M_inv.resize(3 * n);
    for (int i = 0; i < n; ++i) {
        m_iv.M_inv[3 * i + 0] = 1.0 / m_bodies[i]->m;
        m_iv.M_inv[3 * i + 1] = 1.0 / m_bodies[i]->m;
        m_iv.M_inv[3 * i + 2] = 1.0 / m_bodies[i]->I;
    }

    for (int i = 0; i < steps; ++i) {
        m_ode_solver->start(&m_state, dt / steps);

        while (true) {
            const bool done = m_ode_solver->step(&m_state);

#ifdef MANIFOLD_PROFILE
            long long et = 0, st = 0;
            {
                long long ft = 0;
                PROFILE_SCOPE(ft);
                process_forces();
                force_time += ft;
            }
            process_constraints(&et, &st);
            {
                long long ot = 0;
                PROFILE_SCOPE(ot);
                m_ode_solver->solve(&m_state);
                ode_time += ot;
            }
            constraint_time += st;
            eval_time += et;
#else
            process_forces();
            long long et_unused = 0, st_unused = 0;
            process_constraints(&et_unused, &st_unused);
            m_ode_solver->solve(&m_state);
#endif

            if (done)
                break;
        }

        m_ode_solver->end();
    }

    // propagate back to bodies
    for (int i = 0; i < n; ++i) {
        m_bodies[i]->v = m_state.v[i];
        m_bodies[i]->p = m_state.p[i];
        m_bodies[i]->v_theta = m_state.v_theta[i];
        m_bodies[i]->theta = m_state.theta[i];
    }

    // propagate constraint reaction forces
    const int m = get_constraint_count();
    for (int i = 0, i_f = 0; i < m; ++i) {
        Constraint *c = m_constraints[i];
        for (int j = 0; j < c->constraint_count(); ++j, ++i_f) {
            for (int k = 0; k < c->m_body_count; ++k) {
                c->F_xy[j][k] = m_state.r[i_f * 2 + k];
                c->F_t[j][k] = m_state.r_t[i_f * 2 + k];
            }
        }
    }

#ifdef MANIFOLD_PROFILE
    m_prof_ode.push(ode_time);
    m_prof_constraint_solve.push(constraint_time);
    m_prof_constraint_eval.push(eval_time);
    m_prof_force.push(force_time);
#endif
}

void GenericRigidBodySystem::process_constraints(long long *eval_time,
                                                 long long *solve_time) {
#ifdef MANIFOLD_PROFILE
    auto s0 = std::chrono::steady_clock::now();
#endif

    const int n = get_body_count();
    const int m_f = get_full_constraint_count();
    const int m = get_constraint_count();

    // build generalized velocity vector
    m_iv.q_dot.resize(3 * n);
    for (int i = 0; i < n; ++i) {
        m_iv.q_dot[3 * i + 0] = m_state.v[i].x();
        m_iv.q_dot[3 * i + 1] = m_state.v[i].y();
        m_iv.q_dot[3 * i + 2] = m_state.v_theta[i];
    }

    // build external force vector
    m_iv.F_ext.resize(3 * n);
    for (int i = 0; i < n; ++i) {
        m_iv.F_ext[3 * i + 0] = m_state.f[i].x();
        m_iv.F_ext[3 * i + 1] = m_state.f[i].y();
        m_iv.F_ext[3 * i + 2] = m_state.t[i];
    }

    // assemble sparse Jacobian via triplets
    m_iv.C.resize(m_f);
    m_iv.ks.resize(m_f);
    m_iv.kd.resize(m_f);

    std::vector<Triplet<double>> J_triplets, J_dot_triplets;
    J_triplets.reserve(m_f * 6);
    J_dot_triplets.reserve(m_f * 6);

    Constraint::Output output;
    for (int j = 0, j_f = 0; j < m; ++j) {
        m_constraints[j]->calculate(&output, &m_state);
        const int n_f = m_constraints[j]->constraint_count();

        for (int k = 0; k < n_f; ++k, ++j_f) {
            m_iv.C[j_f] = output.C[k];
            m_iv.ks[j_f] = output.ks[k];
            m_iv.kd[j_f] = output.kd[k];

            for (int bi = 0; bi < m_constraints[j]->m_body_count; ++bi) {
                const int idx = m_constraints[j]->m_bodies[bi]->index;
                if (idx == -1)
                    continue;

                for (int d = 0; d < 3; ++d) {
                    double jval = output.J[k][3 * bi + d];
                    double jdval = output.J_dot[k][3 * bi + d];
                    if (jval != 0.0)
                        J_triplets.emplace_back(j_f, 3 * idx + d, jval);
                    if (jdval != 0.0)
                        J_dot_triplets.emplace_back(j_f, 3 * idx + d, jdval);
                }
            }
        }
    }

    m_iv.J.resize(m_f, 3 * n);
    m_iv.J.setFromTriplets(J_triplets.begin(), J_triplets.end());

    m_iv.J_dot.resize(m_f, 3 * n);
    m_iv.J_dot.setFromTriplets(J_dot_triplets.begin(), J_dot_triplets.end());

    // RHS = -(J_dot * q_dot + J * M_inv * F_ext + ks.*C + kd.*(J * q_dot))
    VectorXd Jq = m_iv.J * m_iv.q_dot;
    m_iv.right = -(m_iv.J_dot * m_iv.q_dot +
                   m_iv.J * (m_iv.M_inv.cwiseProduct(m_iv.F_ext)) +
                   m_iv.ks.cwiseProduct(m_iv.C) + m_iv.kd.cwiseProduct(Jq));

#ifdef MANIFOLD_PROFILE
    auto s1 = std::chrono::steady_clock::now();
#endif

    // solve J * M_inv * J^T * lambda = right
    bool solvable = m_sle_solver->solve(m_iv.J, m_iv.M_inv, m_iv.right,
                                        &m_iv.lambda, &m_iv.lambda);
    assert(solvable);

#ifdef MANIFOLD_PROFILE
    auto s2 = std::chrono::steady_clock::now();
#endif

    // compute accelerations: a = M_inv * (F_ext + J^T * lambda)
    VectorXd F_total = m_iv.F_ext + m_iv.J.transpose() * m_iv.lambda;
    VectorXd accel = m_iv.M_inv.cwiseProduct(F_total);

    for (int i = 0; i < n; ++i) {
        m_state.a[i] = Vector2d(accel[3 * i + 0], accel[3 * i + 1]);
        m_state.a_theta[i] = accel[3 * i + 2];
    }

    // store per-constraint reaction forces for visualization
    for (int j = 0, j_f = 0; j < m; ++j) {
        m_constraints[j]->calculate(&output, &m_state);
        const int n_f = m_constraints[j]->constraint_count();

        for (int k = 0; k < n_f; ++k, ++j_f) {
            double lam = m_iv.lambda[j_f];
            for (int bi = 0; bi < m_constraints[j]->m_body_count; ++bi) {
                m_state.r[j_f * 2 + bi] =
                    Vector2d(output.J[k][3 * bi + 0] * lam,
                             output.J[k][3 * bi + 1] * lam);
                m_state.r_t[j_f * 2 + bi] = output.J[k][3 * bi + 2] * lam;
            }
        }
    }

#ifdef MANIFOLD_PROFILE
    auto s3 = std::chrono::steady_clock::now();
    *eval_time =
        std::chrono::duration_cast<std::chrono::microseconds>(s1 - s0 + s3 - s2)
            .count();
    *solve_time =
        std::chrono::duration_cast<std::chrono::microseconds>(s2 - s1).count();
#else
    *eval_time = 0;
    *solve_time = 0;
#endif
}

} // namespace manifold::Solver
