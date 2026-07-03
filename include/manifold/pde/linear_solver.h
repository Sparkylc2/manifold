#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace manifold::PDE {
using namespace Eigen;

// solves J x = b
// general (nonsymmetric) sparse via LU
class LinearSolver {
  public:
    bool solve(const SparseMatrix<double> &J, const VectorXd &b, VectorXd &x);
};

inline bool LinearSolver::solve(const SparseMatrix<double> &J,
                                const VectorXd &b, VectorXd &x) {
    // LU likes compressed column-major
    SparseMatrix<double> A = J;
    A.makeCompressed();

    SparseLU<SparseMatrix<double>> lu;

    lu.compute(A);
    if (lu.info() != Eigen::Success) {
        return false;
    }

    x = lu.solve(b);
    return lu.info() == Eigen::Success;
}

} // namespace manifold::PDE
