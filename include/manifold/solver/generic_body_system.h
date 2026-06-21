#pragma once

#include <manifold/solver/ode_solver.h>
#include <manifold/solver/rigid_body_system.h>
#include <manifold/solver/sle_solver.h>

namespace manifold::Solver {

class GenericRigidBodySystem : public RigidBodySystem {
  public:
    GenericRigidBodySystem();
    ~GenericRigidBodySystem() override = default;

    void initialize(SLESolver *sle_solver, ODESolver *ode_solver);
    void process(double dt, int steps = 1) override;
    void reset() override;

  protected:
    void process_constraints(long long *eval_time, long long *solve_time);

    ODESolver *m_ode_solver = nullptr;
    SLESolver *m_sle_solver = nullptr;

    bool m_is_paused = false;

    struct IntermediateValues {
        SparseMatrix<double> J, J_dot;
        VectorXd M_inv;
        VectorXd C, ks, kd;
        VectorXd q_dot, F_ext;
        VectorXd right;
        VectorXd lambda;
    } m_iv;

    // sparsity pattern caching: avoid triplet rebuild when topology is stable
    bool m_pattern_valid = false;
    int m_cached_n = 0;
    int m_cached_m_f = 0;
};

} // namespace manifold::Solver
