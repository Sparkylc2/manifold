#pragma once

#include <functional>
#include <vector>

#include <Eigen/Sparse>

#include <manifold/pde/grid.h>

namespace manifold::PDE {
using namespace Eigen;

// scalar function of position
using ScalarField = std::function<double(double x, double y)>;

// evaluate g(x, y) at every node to a flat vector
inline VectorXd sample(const Grid &grid, const ScalarField &g) {

    const int N = grid.size();
    VectorXd r = VectorXd::Zero(N);

    for (int k = 0; k < N; k++) {
        int i, j;
        grid.coords(k, i, j);
        r[k] = g(grid.x(i), grid.y(j));
    }

    return r;
}

// a boundary condition overwrites the boundary rows of the solution, the
// residual, and the jacobian. Dirichlet and Neumann implement it differently.
class BoundaryCondition {
  public:
    explicit BoundaryCondition(const Grid &grid) : m_grid(grid) {}
    virtual ~BoundaryCondition() = default;

    virtual void apply_to_solution(VectorXd &u) const = 0;
    virtual void apply_to_residual(const VectorXd &u, VectorXd &F) const = 0;
    virtual void apply_to_jacobian(SparseMatrix<double> &J) const = 0;

  protected:
    const Grid &m_grid;
};

// dirichlet, u = g on the domain edge
class DirichletBC : public BoundaryCondition {
  public:
    DirichletBC(const Grid &grid, ScalarField g)
        : BoundaryCondition(grid), m_g(g) {}

    void apply_to_solution(VectorXd &u) const override;
    void apply_to_residual(const VectorXd &u, VectorXd &F) const override;
    void apply_to_jacobian(SparseMatrix<double> &J) const override;

  private:
    ScalarField m_g;
};

inline void DirichletBC::apply_to_solution(VectorXd &u) const {
    for (int i = 0; i < m_grid.nx(); i++) {
        for (int j = 0; j < m_grid.ny(); j++) {
            if (m_grid.is_boundary(i, j)) {
                const int k = m_grid.idx(i, j);
                u[k] = m_g(m_grid.x(i), m_grid.y(j));
            }
        }
    }
}

inline void DirichletBC::apply_to_residual(const VectorXd &u,
                                           VectorXd &F) const {
    // F[k] = u[k] - g_k
    for (int i = 0; i < m_grid.nx(); i++) {
        for (int j = 0; j < m_grid.ny(); j++) {
            if (m_grid.is_boundary(i, j)) {
                const int k = m_grid.idx(i, j);
                F[k] = u[k] - m_g(m_grid.x(i), m_grid.y(j));
            }
        }
    }
}

inline void DirichletBC::apply_to_jacobian(SparseMatrix<double> &J) const {
    // makes boundary row k the identity row e_k
    const int N = m_grid.size();

    std::vector<bool> on_boundary(N, false);
    for (int i = 0; i < m_grid.nx(); i++)
        for (int j = 0; j < m_grid.ny(); j++)
            if (m_grid.is_boundary(i, j))
                on_boundary[m_grid.idx(i, j)] = true;

    for (int col = 0; col < J.outerSize(); col++)
        for (SparseMatrix<double>::InnerIterator it(J, col); it; ++it)
            if (on_boundary[it.row()])
                it.valueRef() = 0.0;

    for (int k = 0; k < N; k++)
        if (on_boundary[k])
            J.coeffRef(k, k) = 1.0;
}

// neumann, du/dn = q on the domain edge (n = outward normal).
// discretised one-sided: (u_bndry - u_inside)/h = q, where u_inside is the
// neighbour one step along the inward normal. q = 0 is an insulated wall.
// note: an all-Neumann steady problem is singular (solution is only defined up
// to a constant); transient / implicit problems are fine thanks to the I term.
class NeumannBC : public BoundaryCondition {
  public:
    NeumannBC(const Grid &grid, ScalarField q)
        : BoundaryCondition(grid), m_q(q) {}

    void apply_to_solution(VectorXd &u) const override;
    void apply_to_residual(const VectorXd &u, VectorXd &F) const override;
    void apply_to_jacobian(SparseMatrix<double> &J) const override;

  private:
    // step from a boundary node towards the interior (diagonal at corners)
    void inward(int i, int j, int &di, int &dj) const {
        di = (i == 0) ? 1 : (i == m_grid.nx() - 1) ? -1 : 0;
        dj = (j == 0) ? 1 : (j == m_grid.ny() - 1) ? -1 : 0;
    }

    ScalarField m_q;
};

inline void NeumannBC::apply_to_solution(VectorXd &u) const {
    const double h = m_grid.h();
    for (int i = 0; i < m_grid.nx(); i++) {
        for (int j = 0; j < m_grid.ny(); j++) {
            if (!m_grid.is_boundary(i, j))
                continue;
            int di, dj;
            inward(i, j, di, dj);
            const int k = m_grid.idx(i, j);
            const int inside = m_grid.idx(i + di, j + dj);
            u[k] = u[inside] + h * m_q(m_grid.x(i), m_grid.y(j));
        }
    }
}

inline void NeumannBC::apply_to_residual(const VectorXd &u, VectorXd &F) const {
    const double h = m_grid.h();
    for (int i = 0; i < m_grid.nx(); i++) {
        for (int j = 0; j < m_grid.ny(); j++) {
            if (!m_grid.is_boundary(i, j))
                continue;
            int di, dj;
            inward(i, j, di, dj);
            const int k = m_grid.idx(i, j);
            const int inside = m_grid.idx(i + di, j + dj);
            F[k] = (u[k] - u[inside]) / h - m_q(m_grid.x(i), m_grid.y(j));
        }
    }
}

inline void NeumannBC::apply_to_jacobian(SparseMatrix<double> &J) const {
    const int N = m_grid.size();
    const double h = m_grid.h();

    std::vector<bool> on_boundary(N, false);
    for (int i = 0; i < m_grid.nx(); i++)
        for (int j = 0; j < m_grid.ny(); j++)
            if (m_grid.is_boundary(i, j))
                on_boundary[m_grid.idx(i, j)] = true;

    // clear boundary rows
    for (int col = 0; col < J.outerSize(); col++)
        for (SparseMatrix<double>::InnerIterator it(J, col); it; ++it)
            if (on_boundary[it.row()])
                it.valueRef() = 0.0;

    // d(F_k)/du: +1/h at the node, -1/h at the inward neighbour
    for (int i = 0; i < m_grid.nx(); i++) {
        for (int j = 0; j < m_grid.ny(); j++) {
            if (!m_grid.is_boundary(i, j))
                continue;
            int di, dj;
            inward(i, j, di, dj);
            const int k = m_grid.idx(i, j);
            const int inside = m_grid.idx(i + di, j + dj);
            J.coeffRef(k, k) = 1.0 / h;
            J.coeffRef(k, inside) = -1.0 / h;
        }
    }
}

} // namespace manifold::PDE
