#pragma once

#include <manifold/solver/sle_solver.h>

namespace manifold::Solver {

class GaussianEliminationSLESolver : public SLESolver {
  public:
    GaussianEliminationSLESolver() : SLESolver(false) {}

    bool solve(SparseMatrix<double> &J, VectorXd &W, VectorXd &right,
               VectorXd *result, VectorXd *previous) override;
};

} // namespace manifold::Solver
