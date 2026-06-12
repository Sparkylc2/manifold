#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>

namespace manifold::Solver {
using namespace Eigen;
class SLESolver {
  public:
    SLESolver(bool supports_limits);
    virtual ~SLESolver() = default;

    virtual bool solve(SparseMatrix<double> &J, VectorXd &W, VectorXd &right,
                       VectorXd *result, VectorXd *previous);

    virtual bool solve_with_limits(SparseMatrix<double> &J, VectorXd &W,
                                   VectorXd &right, VectorXd &limits,
                                   VectorXd *result, VectorXd *previous);

    bool supports_limits() const { return m_supports_limits; }

  private:
    bool m_supports_limits;
};
} // namespace manifold::Solver
