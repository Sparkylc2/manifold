#pragma once

#include <manifold/pde/boundary_condition.h>
#include <manifold/pde/operator.h>

namespace manifold::PDE {
using namespace Eigen;

// bundles an operator L, source term f, and bc's into a system
// F(u) = 0
class Problem {
  public:
    Problem(const Grid &grid, const Operator &op, VectorXd f,
            const BoundaryCondition &bc)
        : m_grid(grid), m_op(op), m_f(std::move(f)), m_bc(bc) {}

    // F(u) = L[u] - f, boundary rows overwritten by the BC
    VectorXd residual(const VectorXd &u) const;

    // dF/du, with boundary rows overwritten by the BC
    SparseMatrix<double> jacobian(const VectorXd &u) const;

    const Grid &grid() const { return m_grid; }
    const BoundaryCondition &bc() const { return m_bc; }

  private:
    const Grid &m_grid;
    const Operator &m_op;
    VectorXd m_f;
    const BoundaryCondition &m_bc;
};

inline VectorXd Problem::residual(const VectorXd &u) const {
    VectorXd F = m_op.eval(u) - m_f;
    m_bc.apply_to_residual(u, F);
    return F;
}

inline SparseMatrix<double> Problem::jacobian(const VectorXd &u) const {
    SparseMatrix<double> J = m_op.jacobian(u);
    m_bc.apply_to_jacobian(J);
    return J;
}

} // namespace manifold::PDE
