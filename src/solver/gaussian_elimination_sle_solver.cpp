#include <manifold/solver/gaussian_elimination_sle_solver.h>

namespace manifold::Solver {

bool GaussianEliminationSLESolver::solve(SparseMatrix<double> &J, VectorXd &W,
                                         VectorXd &right, VectorXd *result,
                                         VectorXd *previous) {

    // form A = J * diag(W) * J^T
    MatrixXd A = MatrixXd(J) * W.asDiagonal() * MatrixXd(J).transpose();
    A.diagonal().array() += 1e-6; // Tikhonov regularization

    // direct solve via partial-pivot LU
    *result = A.partialPivLu().solve(right);
    return true;
}
} // namespace manifold::Solver
