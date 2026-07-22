#include <manifold/fea/boundary.h>

#include <stdexcept>

namespace manifold::FEA {

void BoundaryConditions::clear() {
    m_dirichlet.clear();
    m_neumann.clear();
    m_robin.clear();
}

void BoundaryConditions::clear_neumann() { m_neumann.clear(); }

void BoundaryConditions::clear_robin() { m_robin.clear(); }

void BoundaryConditions::add_dirichlet(int dof, double value) {
    for (Dirichlet &d : m_dirichlet) {
        if (d.dof == dof) {
            d.value = value;
            return;
        }
    }
    m_dirichlet.push_back({dof, value});
}

void BoundaryConditions::remove_dirichlet(int dof) {
    for (size_t i = 0; i < m_dirichlet.size(); i++) {
        if (m_dirichlet[i].dof == dof) {
            m_dirichlet.erase(m_dirichlet.begin() + (long)i);
            return;
        }
    }
}

void BoundaryConditions::add_neumann(int dof, double value) {
    m_neumann.push_back({dof, value});
}

void BoundaryConditions::add_robin(const RobinEdge &e) { m_robin.push_back(e); }

void BoundaryConditions::apply_loads(VectorXd &rhs) const {
    for (const Neumann &n : m_neumann)
        rhs(n.dof) += n.value;

    // q += h*L/2 * t_inf on each of the edge's two nodes
    for (const RobinEdge &e : m_robin) {
        const double v = 0.5 * e.h * e.length * e.t_inf;
        rhs(e.dof_a) += v;
        rhs(e.dof_b) += v;
    }
}

void BoundaryConditions::apply_robin_stiffness(SparseMatrix<double> &K) const {
    // Kc += h*L/6 * [[2,1],[1,2]]
    for (const RobinEdge &e : m_robin) {
        const double s = e.h * e.length / 6.0;
        K.coeffRef(e.dof_a, e.dof_a) += 2.0 * s;
        K.coeffRef(e.dof_a, e.dof_b) += 1.0 * s;
        K.coeffRef(e.dof_b, e.dof_a) += 1.0 * s;
        K.coeffRef(e.dof_b, e.dof_b) += 2.0 * s;
    }
}

void BoundaryConditions::fixed_mask(int n, std::vector<char> &mask) const {
    mask.assign((size_t)n, 0);
    for (const Dirichlet &d : m_dirichlet) {
        if (d.dof < 0 || d.dof >= n)
            throw std::out_of_range("dirichlet dof out of range");
        mask[(size_t)d.dof] = 1;
    }
}

void BoundaryConditions::apply_dirichlet(SparseMatrix<double> &K,
                                         VectorXd &rhs) const {
    if (m_dirichlet.empty())
        return;

    const int n = (int)rhs.size();
    std::vector<char> fixed;
    std::vector<double> value((size_t)n, 0.0);
    fixed_mask(n, fixed);

    for (const Dirichlet &d : m_dirichlet)
        value[(size_t)d.dof] = d.value;

    // move the known columns to the rhs before dropping them, this keeps the
    // reduced system symmetric
    for (int col = 0; col < K.outerSize(); col++) {
        if (!fixed[(size_t)col])
            continue;

        for (SparseMatrix<double>::InnerIterator it(K, col); it; ++it) {
            const int row = it.row();
            if (!fixed[(size_t)row])
                rhs(row) -= it.value() * value[(size_t)col];
        }
    }

    for (int col = 0; col < K.outerSize(); col++) {
        for (SparseMatrix<double>::InnerIterator it(K, col); it; ++it) {
            const int row = it.row();
            if (fixed[(size_t)row] || fixed[(size_t)col])
                it.valueRef() = (row == col) ? 1.0 : 0.0;
        }
    }

    for (const Dirichlet &d : m_dirichlet) {
        K.coeffRef(d.dof, d.dof) = 1.0;
        rhs(d.dof) = d.value;
    }
}

} // namespace manifold::FEA
