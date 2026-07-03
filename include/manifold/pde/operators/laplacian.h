#pragma once

#include <manifold/pde/operator.h>

namespace manifold::PDE {

// L[u] = coeff * laplacian(u)
// only interior rows are filled
class Laplacian : public Operator {
  public:
    Laplacian(const Grid &grid, double coeff = 1.0)
        : Operator(grid), m_coeff(coeff) {}

    VectorXd eval(const VectorXd &u) const override;
    SparseMatrix<double> jacobian(const VectorXd &u) const override;

  private:
    double m_coeff;
};

inline VectorXd Laplacian::eval(const VectorXd &u) const {
    // laplacian at node (i, j) (5 point)
    // (u_E + u_W + u_N + u_S - 4 * u_P) / h^2

    const double h2_r = 1.0 / (m_grid.h() * m_grid.h());
    VectorXd r = VectorXd::Zero(m_grid.size());

    for (int i = 1; i < m_grid.nx() - 1; i++) {
        for (int j = 1; j < m_grid.ny() - 1; j++) {

            const int k = m_grid.idx(i, j);

            const double u_P = u[k];

            const double u_E = u[m_grid.idx(i + 1, j)];
            const double u_W = u[m_grid.idx(i - 1, j)];
            const double u_N = u[m_grid.idx(i, j + 1)];
            const double u_S = u[m_grid.idx(i, j - 1)];

            r[k] = m_coeff * (u_E + u_W + u_N + u_S - 4 * u_P) * h2_r;
        }
    }

    return r;
}

inline SparseMatrix<double> Laplacian::jacobian(const VectorXd &u) const {

    //  simple derivatives for each point and its neighbours

    const int N = m_grid.size();

    std::vector<Triplet<double>> J_vals;

    // common factor of the derivatives
    const double c = m_coeff * 1.0 / (m_grid.h() * m_grid.h());

    for (int i = 1; i < m_grid.nx() - 1; i++) {
        for (int j = 1; j < m_grid.ny() - 1; j++) {
            const int k = m_grid.idx(i, j);

            // d/du(-4 * m_coeff / h^2 * u_P)
            J_vals.emplace_back(k, k, -4 * c);
            // d/du(m_coeff / h^2 * u_E)
            J_vals.emplace_back(k, m_grid.idx(i + 1, j), c);
            // you get the idea
            J_vals.emplace_back(k, m_grid.idx(i - 1, j), c);
            J_vals.emplace_back(k, m_grid.idx(i, j + 1), c);
            J_vals.emplace_back(k, m_grid.idx(i, j - 1), c);
        }
    }

    SparseMatrix<double> J(N, N);
    J.setFromTriplets(J_vals.begin(), J_vals.end());
    return J;
}

} // namespace manifold::PDE
