#pragma once
#include <Eigen/Dense>
#include <vector>

namespace manifold::FEA {
using namespace Eigen;

// generic finite element
struct Element {
    virtual ~Element() = default;

    // identification and sizing
    virtual int num_nodes() const = 0;
    virtual int dofs_per_node() const = 0;

    // total dofs for the element
    int total_dofs() const { return num_nodes() * dofs_per_node(); }
    // the global nodes that make up the element
    virtual const std::vector<int> &node_ids() const = 0;

    // local matrices for global assembly (node-major dof order)
    virtual void local_stiffness(MatrixXd &K) const = 0;
    virtual void local_mass(MatrixXd &M) const = 0;
};
} // namespace manifold::FEA
