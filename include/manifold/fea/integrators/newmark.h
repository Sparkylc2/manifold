#pragma once

#include <Eigen/Sparse>
#include <manifold/solver/conjugate_gradient_sle_solver.h>

namespace manifold::FEA {
using namespace Eigen;

// implicit Newmark-beta for
//     M*a + C*v + f_int(u) = f_ext
//
// given tangent K_eff and internal force f_int at the
// current/predicted configuration, one step solves a_{n+1} from:
//      (M + gamma*dt*C + beta*dt^2*K_eff) a_{n+1} = f_ext - C v* - f_int
// where
//   u* = u_n + dt v_n + dt^2 (1/2 - beta) a_n
//   v* = v_n + dt (1 - gamma) a_n
// then corrects
//   u_{n+1} = u* + beta*dt^2*a_{n+1}
//   v_{n+1} = v* + gamma*dt*a_{n+1}
//
// K_eff and f_int are assembled by the ElasticBody
class Newmark {
  public:
    Newmark(double beta = 0.25, double gamma = 0.5)
        : m_beta(beta), m_gamma(gamma) {}

    // state vectors (global dofs)
    struct State {
        VectorXd u; // displacement (p - x)
        VectorXd v; // velocity
        VectorXd a; // acceleration
    };

    // K_eff = warped tangent
    // f_int = warped internal force at current u
    // f_ext = external loads (Dirichlet already applied to K_eff/rhs)
    // f_int and K_eff come from the configuration used for this step
    void step(double dt, const SparseMatrix<double> &M,
              const SparseMatrix<double> &C, const SparseMatrix<double> &K_eff,
              const VectorXd &f_int, const VectorXd &f_ext, State &s);

    // the caller has to build f_int/K_eff at the same predictor this uses
    double beta() const { return m_beta; }
    double gamma() const { return m_gamma; }

    void set_cg_max_iter(int v) { m_cg_max_iter = v; }
    void set_cg_rel_tol(double v) { m_cg_rel_tol = v; }
    void set_cg_abs_tol(double v) { m_cg_abs_tol = v; }

    int cg_max_iter() const { return m_cg_max_iter; }
    double cg_rel_tol() const { return m_cg_rel_tol; }
    double cg_abs_tol() const { return m_cg_abs_tol; }

  private:
    double m_beta;
    double m_gamma;
    int m_cg_max_iter = 300;
    double m_cg_rel_tol = 1e-6;
    double m_cg_abs_tol = 1e-12;

    // scratch stuff
    SparseMatrix<double> m_A;
    VectorXd m_x;
    VectorXd m_rhs;
    VectorXd m_r;
    VectorXd m_p;
    VectorXd m_Ap;
    VectorXd m_u_star;
    VectorXd m_v_star;
    int m_last_n = -1;
};

} // namespace manifold::FEA
