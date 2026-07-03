#pragma once

#include <manifold/pde/operator.h>

namespace manifold::PDE {

// L[u] = v dot grad(u)
// first-order upwind
class Gradient : public Operator {
  public:
    Gradient(const Grid &grid, double vx, double vy)
        : Operator(grid), m_vx(vx), m_vy(vy) {}

    VectorXd eval(const VectorXd &u) const override;
    SparseMatrix<double> jacobian(const VectorXd &u) const override;

  private:
    double m_vx, m_vy;
};

inline VectorXd Gradient::eval(const VectorXd &u) const {

    const int N = m_grid.size();
    VectorXd r = VectorXd::Zero(N);

    for (int i = 1; i < m_grid.nx() - 1; i++) {
        for (int j = 1; j < m_grid.ny() - 1; j++) {
            const int k = m_grid.idx(i, j);

            double d_u_x, d_u_y;

            // upwind branches
            if (m_vx > 0) {
                // (u_P - u_W) / h
                d_u_x = (u[k] - u[m_grid.idx(i - 1, j)]) / m_grid.h();
            } else {
                // (u_E - u_P) / h
                d_u_x = (u[m_grid.idx(i + 1, j)] - u[k]) / m_grid.h();
            }

            if (m_vy > 0) {
                // (u_P - u_S) / h
                d_u_y = (u[k] - u[m_grid.idx(i, j - 1)]) / m_grid.h();
            } else {
                // (u_N - u_P) / h
                d_u_y = (u[m_grid.idx(i, j + 1)] - u[k]) / m_grid.h();
            }

            r[k] = m_vx * d_u_x + m_vy * d_u_y;
        }
    }
    return r;
}

inline SparseMatrix<double> Gradient::jacobian(const VectorXd &u) const {

    const int N = m_grid.size();
    std::vector<Triplet<double>> J_vals;

    const double vx_h = m_vx / m_grid.h();
    const double vy_h = m_vy / m_grid.h();

    for (int i = 1; i < m_grid.nx() - 1; i++) {
        for (int j = 1; j < m_grid.ny() - 1; j++) {
            const int k = m_grid.idx(i, j);
            if (m_vx > 0) {
                // (k, k, vx/h)
                // (k, W, -vx/h)
                J_vals.emplace_back(k, k, vx_h);
                J_vals.emplace_back(k, m_grid.idx(i - 1, j), -vx_h);
            } else {
                // (k, k, -vx/h)
                // (k, E, vx_h)
                J_vals.emplace_back(k, k, -vx_h);
                J_vals.emplace_back(k, m_grid.idx(i + 1, j), vx_h);
            }

            if (m_vy > 0) {
                // (k, k, vy_h)
                // (k, S, -vy_h)
                J_vals.emplace_back(k, k, vy_h);
                J_vals.emplace_back(k, m_grid.idx(i, j - 1), -vy_h);
            } else {
                // (k, k, -vy_h)
                // (k, N, vy_h)
                J_vals.emplace_back(k, k, -vy_h);
                J_vals.emplace_back(k, m_grid.idx(i, j + 1), vy_h);
            }
        }
    }

    SparseMatrix<double> J(N, N);
    J.setFromTriplets(J_vals.begin(), J_vals.end());
    return J;
}

} // namespace manifold::PDE
