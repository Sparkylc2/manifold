#pragma once

#include <manifold/pde/operator.h>

namespace manifold::PDE {

// L[u] = u .* inner[u].
// nonlinear elementwise product:
// inner = d/dx gives u * u_x
class Multiply : public Operator {
  public:
    Multiply(const Grid &grid, const Operator &inner)
        : Operator(grid), m_inner(inner) {}

    VectorXd eval(const VectorXd &u) const override;

  private:
    const Operator &m_inner;
};

inline VectorXd Multiply::eval(const VectorXd &u) const {
    return u.cwiseProduct(m_inner.eval(u));
}

} // namespace manifold::PDE
