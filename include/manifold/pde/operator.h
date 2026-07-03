#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseCore>

#include <manifold/pde/grid.h>

namespace manifold::PDE {
using namespace Eigen;

// a differential operator L acting on grid values u
// residual(u) returns L[u]
// jacobian(u) returns d(L[u])/du.
// linear operators override jacobian() with the exact constant matrix
// nonlinear ones use finite-difference
class Operator {
  public:
    explicit Operator(const Grid &grid) : m_grid(grid) {}
    virtual ~Operator() = default;

    virtual VectorXd eval(const VectorXd &u) const = 0;
    virtual SparseMatrix<double> jacobian(const VectorXd &u) const;

  protected:
    const Grid &m_grid;

    SparseMatrix<double> finite_difference_jacobian(const VectorXd &u,
                                                    double eps = 1e-7) const;
};

inline SparseMatrix<double> Operator::jacobian(const VectorXd &u) const {
    return finite_difference_jacobian(u);
}

inline SparseMatrix<double>
Operator::finite_difference_jacobian(const VectorXd &u, double eps) const {

    const int N = m_grid.size();
    VectorXd F0 = this->eval(u);

    // working copy
    VectorXd perturbed = u;
    std::vector<Triplet<double>> J_vals{};

    // the j'th column of the jacobian is how the whole residual vector changes
    // if we nudge an unknown u_j
    // we build this to solve the linearized form of the equations
    // F(u + du) = F(u) + J*du
    //
    // iterating over all columns
    for (int j = 0; j < N; j++) {

        perturbed[j] += eps;                 // nudging u_j
        VectorXd F1 = this->eval(perturbed); // perturbed result
        perturbed[j] = u[j];                 // un-nudging u_j

        VectorXd col = (F1 - F0) / eps; // derivative

        // our matrix is sparse so we can skip most entries
        for (int i = 0; i < N; i++) {
            if (std::abs(col[i]) > 1e-12) {
                J_vals.emplace_back(i, j, col[i]);
            }
        }
    }

    // build J
    SparseMatrix<double> J(N, N);
    J.setFromTriplets(J_vals.begin(), J_vals.end());
    return J;
}

} // namespace manifold::PDE
