#pragma once

#include <manifold/fea/element.h>

#include <Eigen/Sparse>
#include <vector>

namespace manifold::FEA {
using namespace Eigen;

// scatters per-element local matrices into the global system
class Assembler {
  public:
    Assembler(int node_count, int dofs_per_node)
        : m_nodes(node_count), m_dpn(dofs_per_node) {}

    int dof_count() const { return m_nodes * m_dpn; }
    int dof_of(int node, int comp) const { return node * m_dpn + comp; }

    // assembles global K (stiffness/conduction) from local_stiffness of each
    void assemble_stiffness(const std::vector<Element *> &elems,
                            SparseMatrix<double> &K) const;

    // assembles global M (mass/capacity) from local_mass of each
    void assemble_mass(const std::vector<Element *> &elems,
                       SparseMatrix<double> &M) const;

    // pushes one local (total_dofs x total_dofs) block into the triplet list at
    // the element's global dof rows/cols
    void scatter(const Element &e, const MatrixXd &local,
                 std::vector<Triplet<double>> &out) const;

    // same mapping for a local dof vector, ie a warped element force
    void scatter_vector(const Element &e, const VectorXd &local,
                        VectorXd &out) const;

  private:
    int m_nodes;
    int m_dpn;
};

} // namespace manifold::FEA
