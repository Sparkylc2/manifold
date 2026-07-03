#pragma once

#include <manifold/pde/operator.h>

namespace manifold::PDE {

// L[u] = alpha * inner[u]
// scales another operator
class Scale : public Operator {
  public:
    Scale(const Grid &grid, const Operator &inner, double alpha)
        : Operator(grid), m_inner(inner), m_alpha(alpha) {}

    VectorXd eval(const VectorXd &u) const override;
    SparseMatrix<double> jacobian(const VectorXd &u) const override;

  private:
    const Operator &m_inner;
    double m_alpha;
};

inline VectorXd Scale::eval(const VectorXd &u) const {
    return m_alpha * m_inner.eval(u);
}

inline SparseMatrix<double> Scale::jacobian(const VectorXd &u) const {
    return m_alpha * m_inner.jacobian(u);
}

} // namespace manifold::PDE
