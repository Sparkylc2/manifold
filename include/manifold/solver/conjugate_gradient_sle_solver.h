#pragma once

#include <manifold/solver/sle_solver.h>

namespace manifold::Solver {
class ConjugateGradientSLESolver : public SLESolver {
  public:
    ConjugateGradientSLESolver();
    virtual ~ConjugateGradientSLESolver() = default;

    bool solve(SparseMatrix<double> &J, VectorXd &W, VectorXd &right,
               VectorXd *result, VectorXd *previous);

    void set_max_iter(int max_iter) { m_max_iter = max_iter; }
    int get_max_iter() const { return m_max_iter; }

    void set_max_err(double max_err) { m_max_err = max_err; }
    double get_max_err() const { return m_max_err; }

    void set_min_err(double min_err) { m_min_err = min_err; }
    double get_min_err() const { return m_min_err; }

  protected:
    void multiply(SparseMatrix<double> &J, VectorXd &W, VectorXd &x,
                  VectorXd *target);
    bool sufficiently_small(const VectorXd &x, const VectorXd &target) const;

    // CG working vectors
    VectorXd m_r, m_p, m_Ap, m_x;

    // multiply scratch (sized to DOF count, not constraint count)
    VectorXd m_mreg0, m_mreg1;

    // cached explicit transpose of J (avoids view overhead per CG iteration)
    SparseMatrix<double> m_J_T;

    // pre-allocation tracking — only resize when dimensions change
    int m_last_n = -1;
    int m_last_n_dof = -1;

    int m_max_iter = 200;
    double m_max_err = 1e-6;
    double m_min_err = 1e-12;
};
} // namespace manifold::Solver
