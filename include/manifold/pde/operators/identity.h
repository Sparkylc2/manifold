#pragma once

#include <manifold/pde/operator.h>

namespace manifold::PDE {

// L[u] = c * u
class Identity : public Operator {
  public:
    Identity(const Grid &grid, double c = 1.0) : Operator(grid), m_c(c) {}

    VectorXd eval(const VectorXd &u) const override;
    SparseMatrix<double> jacobian(const VectorXd &u) const override;

  private:
    double m_c;
};

inline VectorXd Identity::eval(const VectorXd &u) const { return m_c * u; }

inline SparseMatrix<double> Identity::jacobian(const VectorXd &u) const {

    const int N = m_grid.size();
    std::vector<Triplet<double>> J_vals;

    // d(c * u) / du = c * I
    for (int k = 0; k < N; k++) {
        J_vals.emplace_back(k, k, m_c);
    }

    SparseMatrix<double> J(N, N);
    J.setFromTriplets(J_vals.begin(), J_vals.end());
    return J;
}

} // namespace manifold::PDE
