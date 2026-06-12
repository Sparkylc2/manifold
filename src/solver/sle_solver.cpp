#include <manifold/solver/sle_solver.h>

namespace manifold::Solver {

SLESolver::SLESolver(bool supports_limits) {
    m_supports_limits = supports_limits;
}

bool SLESolver::solve(SparseMatrix<double> &J, VectorXd &W, VectorXd &right,
                      VectorXd *result, VectorXd *previous) {
    return false;
}

bool SLESolver::solve_with_limits(SparseMatrix<double> &J, VectorXd &W,
                                  VectorXd &right, VectorXd &limits,
                                  VectorXd *result, VectorXd *previous) {
    return false;
}
} // namespace manifold::Solver
