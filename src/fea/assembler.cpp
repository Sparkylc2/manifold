#include <manifold/fea/assembler.h>
#include <stdexcept>

namespace manifold::FEA {

using namespace Eigen;

void Assembler::scatter(const Element &e, const MatrixXd &local,
                        std::vector<Triplet<double>> &out) const {

    const auto &ids = e.node_ids();
    const int nnode = e.num_nodes();
    const int dpn = e.dofs_per_node();
    const int ldof = nnode * dpn;

    if (ids.size() != nnode)
        throw std::invalid_argument("element node_ids size mismatch");

    if (dpn != m_dpn)
        throw std::invalid_argument("dofs_per_node mismatch with assembler");

    if (local.rows() != ldof || local.cols() != ldof)
        throw std::invalid_argument("local matrix dimension mismatch");

    for (int a = 0; a < nnode; a++) {
        const int node_a = ids[a];
        if (node_a < 0 || node_a >= m_nodes)
            throw std::out_of_range("element node index out of range");

        for (int ca = 0; ca < dpn; ca++) {
            const int row_local = a * dpn + ca;
            const int row_global = dof_of(node_a, ca);

            for (int b = 0; b < nnode; ++b) {
                const int node_b = ids[b];
                if (node_b < 0 || node_b >= m_nodes)
                    throw std::out_of_range("element node index out of range");

                for (int cb = 0; cb < dpn; ++cb) {
                    const int col_local = b * dpn + cb;
                    const int col_global = dof_of(node_b, cb);
                    const double v = local(row_local, col_local);
                    if (v != 0.0)
                        out.emplace_back(row_global, col_global, v);
                }
            }
        }
    }
}

void Assembler::scatter_vector(const Element &e, const VectorXd &local,
                               VectorXd &out) const {

    const auto &ids = e.node_ids();
    const int nnode = e.num_nodes();
    const int dpn = e.dofs_per_node();

    if (dpn != m_dpn)
        throw std::invalid_argument("dofs_per_node mismatch with assembler");

    if (local.size() != nnode * dpn)
        throw std::invalid_argument("local vector dimension mismatch");

    for (int a = 0; a < nnode; a++) {
        const int node_a = ids[a];
        if (node_a < 0 || node_a >= m_nodes)
            throw std::out_of_range("element node index out of range");

        for (int ca = 0; ca < dpn; ca++)
            out(dof_of(node_a, ca)) += local(a * dpn + ca);
    }
}

void Assembler::assemble_stiffness(const std::vector<Element *> &elems,
                                   SparseMatrix<double> &K) const {

    std::vector<Triplet<double>> triplets;
    triplets.reserve(elems.size() * 64);

    for (const Element *e : elems) {
        if (!e) {
            throw std::invalid_argument(
                "Assembler::assemble_stiffness received null element");
        }

        MatrixXd local;
        e->local_stiffness(local);
        scatter(*e, local, triplets);
    }

    K.setFromTriplets(triplets.begin(), triplets.end(),
                      [](const double &a, const double &b) { return a + b; });
    K.makeCompressed();
}

void Assembler::assemble_mass(const std::vector<Element *> &elems,
                              SparseMatrix<double> &M) const {

    std::vector<Triplet<double>> triplets;
    triplets.reserve(elems.size() * 64);

    for (const Element *e : elems) {
        if (!e) {
            throw std::invalid_argument(
                "Assembler::assemble_mass received null element");
        }

        MatrixXd local;
        e->local_mass(local);
        scatter(*e, local, triplets);
    }

    M.setFromTriplets(triplets.begin(), triplets.end(),
                      [](const double &a, const double &b) { return a + b; });
    M.makeCompressed();
}

} // namespace manifold::FEA
