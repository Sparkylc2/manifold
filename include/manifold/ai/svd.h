#pragma once
#include <Eigen/Core>

namespace manifold::AI {
using namespace Eigen;

// stores the X = U Sigma V info
struct SVDResult {
    MatrixXd U;
    VectorXd S;
    MatrixXd V;
};

// one-sided Jacobi SVD (basic implementation, snapshots would be better)
SVDResult jacobi_svd(const MatrixXd &A, double tol = 1e-14, int max_sweeps = 60);

// thin SVD via Eigen's BDCSVD
SVDResult eigen_svd(const MatrixXd &A);

} // namespace manifold::AI
