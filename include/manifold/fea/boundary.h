#pragma once

#include <Eigen/Sparse>
#include <vector>

namespace manifold::FEA {
using namespace Eigen;

// boundary conditions for both physics, dof indices are global (from Assembler)

// prescribed dof value, ie a fixed support or a node dragged by a cart.
// couplers retarget `value` each step
struct Dirichlet {
    int dof = -1;
    double value = 0.0;
};

// external load added to the rhs, ie a nodal force or an integrated
// traction/flux
struct Neumann {
    int dof = -1;
    double value = 0.0;
};

// convection on a wetted edge, applied as an edge integral
//      Kc += h*L/6 * [[2,1],[1,2]]
//      q  += h*L/2 * t_inf * [1,1]
struct RobinEdge {
    int dof_a = -1;
    int dof_b = -1;
    double length = 0.0; // rest edge length
    double h = 0.0;      // film coefficient, may track fluid speed
    double t_inf = 0.0;  // ambient/fluid temperature
};

class BoundaryConditions {
  public:
    void clear();
    void clear_neumann();
    void clear_robin();

    // upserts, so a coupler can retarget the same dof every step
    void add_dirichlet(int dof, double value);
    void remove_dirichlet(int dof);

    void add_neumann(int dof, double value);
    void add_robin(const RobinEdge &e);

    // folds neumann + robin loads into the rhs
    void apply_loads(VectorXd &rhs) const;

    // folds robin edge conductance into K, before the solve
    void apply_robin_stiffness(SparseMatrix<double> &K) const;

    // symmetric row/col elimination of the dirichlet dofs on (K, rhs)
    void apply_dirichlet(SparseMatrix<double> &K, VectorXd &rhs) const;

    // marks which dofs are dirichlet, used when a caller has to eliminate
    // across several matrices at once (see ElasticBody::advance)
    void fixed_mask(int n, std::vector<char> &mask) const;

    const std::vector<Dirichlet> &dirichlet() const { return m_dirichlet; }
    const std::vector<RobinEdge> &robin() const { return m_robin; }

  private:
    std::vector<Dirichlet> m_dirichlet;
    std::vector<Neumann> m_neumann;
    std::vector<RobinEdge> m_robin;
};

} // namespace manifold::FEA
