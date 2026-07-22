#pragma once

#include <Eigen/Sparse>

namespace manifold::FEA {
using namespace Eigen;

// implicit (backward) euler for the first-order heat system
//      C*T' + Kc*T = q
// one step solves
//      (C + dt*Kc) T_{n+1} = C*T_n + dt*q
//
// unconditionally stable, first order in time is plenty for conduction.
// dirichlet (fixed temperature) and robin (convection) are folded into
// Kc / q by BoundaryConditions before this is called
class BackwardEuler {
  public:
    // returns T_{n+1} in T
    void step(double dt, const SparseMatrix<double> &C,
              const SparseMatrix<double> &Kc, const VectorXd &q, VectorXd &T);

    void set_cg_max_iter(int v) { m_cg_max_iter = v; }
    void set_cg_rel_tol(double v) { m_cg_rel_tol = v; }
    void set_cg_abs_tol(double v) { m_cg_abs_tol = v; }

  private:
    int m_cg_max_iter = 300;
    double m_cg_rel_tol = 1e-8;
    double m_cg_abs_tol = 1e-14;

    // scratch stuff
    SparseMatrix<double> m_A;
    VectorXd m_rhs;
    VectorXd m_r;
    VectorXd m_p;
    VectorXd m_Ap;
    int m_last_n = -1;
};

} // namespace manifold::FEA
