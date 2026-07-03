#pragma once

#include <vector>

#include <manifold/pde/operator.h>

namespace manifold::PDE {

// L[u] = sum_k term_k[u]
// sum of operators
class Sum : public Operator {
  public:
    explicit Sum(const Grid &grid) : Operator(grid) {}

    void add(const Operator &term) { m_terms.push_back(&term); }

    VectorXd eval(const VectorXd &u) const override;
    SparseMatrix<double> jacobian(const VectorXd &u) const override;

  private:
    std::vector<const Operator *> m_terms;
};

inline VectorXd Sum::eval(const VectorXd &u) const {

    const int N = m_grid.size();
    VectorXd r = VectorXd::Zero(N);

    // accumulating
    for (const Operator *t : m_terms) {
        r += t->eval(u);
    }

    return r;
}

inline SparseMatrix<double> Sum::jacobian(const VectorXd &u) const {

    const int N = m_grid.size();
    SparseMatrix<double> J(N, N);

    // accumulating
    for (const Operator *t : m_terms) {
        J += t->jacobian(u);
    }
    return J;
}

} // namespace manifold::PDE
