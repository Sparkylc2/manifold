#pragma once

#include <manifold/fea/assembler.h>
#include <manifold/fea/boundary.h>
#include <manifold/fea/element.h>
#include <manifold/fea/elements/cst_elastic.h>
#include <manifold/fea/integrators/backward_euler.h>
#include <manifold/fea/integrators/newmark.h>
#include <manifold/fea/material.h>
#include <manifold/fea/mesh.h>

#include <Eigen/Sparse>
#include <memory>
#include <vector>

namespace manifold::FEA {
using namespace Eigen;

// mechanical FEA domain, mirrors Fluid::FluidSolver
// advance by a shared dt, expose node state to couplers, accept nodal loads.
// this is the surface the coupling layer sees
class FeaSolver {
  public:
    virtual ~FeaSolver() = default;

    virtual void advance(double dt) = 0;

    virtual int node_count() const = 0;
    virtual Vector2d node_position(int i) const = 0;
    virtual Vector2d node_velocity(int i) const = 0;

    // cleared each tick by the driver before couplers re-add
    virtual void add_nodal_force(int i, const Vector2d &f) = 0;
    virtual void clear_loads() = 0;

    // dirichlet node a coupler retargets each step, ie a cart mount
    virtual void set_fixed_node(int i, const Vector2d &target_world) = 0;
    virtual void release_node(int i) = 0;

    virtual double von_mises(int element) const = 0;
};

// mesh of CstElastic, corotational tangent + implicit newmark
class ElasticBody : public FeaSolver {
  public:
    ElasticBody(Mesh mesh, const Material &mat);

    // builds the elements and the constant M, K0, C. call once
    void build();
    void reset();

    void advance(double dt) override;

    int node_count() const override { return m_mesh.node_count(); }
    Vector2d node_position(int i) const override;
    Vector2d node_velocity(int i) const override;

    void add_nodal_force(int i, const Vector2d &f) override;
    void clear_loads() override { m_fext.setZero(); }

    void set_fixed_node(int i, const Vector2d &target_world) override;
    void release_node(int i) override;

    double von_mises(int element) const override;

    int element_count() const { return (int)m_elems.size(); }
    const Mesh &mesh() const { return m_mesh; }
    const Material &material() const { return m_mat; }
    Newmark &integrator() { return m_integrator; }

    // total elastic + kinetic energy, handy as a stability check
    double energy() const;

  private:
    // warped tangent sum(Re*Ke*Re^T) and internal force sum(Re*Ke*(Re^T*p - x)),
    // both evaluated at the displacement u
    void assemble_warped(const VectorXd &u, SparseMatrix<double> &K_eff,
                         VectorXd &f_int);

    // world coords of element e's nodes at displacement u, node-major
    void gather_positions(int e, const VectorXd &u, VectorXd &pos) const;

    Mesh m_mesh;
    Material m_mat;
    std::vector<std::unique_ptr<Element>> m_elems;
    std::vector<Element *> m_elem_ptrs;
    std::vector<CstElastic *> m_cst;

    std::unique_ptr<Assembler> m_assembler;
    Newmark m_integrator;

    SparseMatrix<double> m_M;  // mass
    SparseMatrix<double> m_K0; // unwarped stiffness, only for rayleigh damping
    SparseMatrix<double> m_C;  // C = alpha*M + beta*K0

    Newmark::State m_state; // u, v, a over global dofs
    VectorXd m_fext;

    std::vector<int> m_fixed_node;
    std::vector<Vector2d> m_fixed_target;

    // scratch stuff
    SparseMatrix<double> m_K_eff;
    SparseMatrix<double> m_M_bc;
    SparseMatrix<double> m_C_bc;
    VectorXd m_f_int;
    VectorXd m_f_ext_bc;
    VectorXd m_u_pred;
    std::vector<char> m_fixed_dof;
    bool m_built = false;
};

// thermal field on the same mesh, TriThermal + backward euler.
// shares the mesh/assembler machinery with ElasticBody, separate because the
// dof layout is scalar (1/node) rather than vector (2/node)
class ThermalBody {
  public:
    ThermalBody(Mesh mesh, const Material &mat);

    void build();
    void set_uniform(double T0);

    void advance(double dt);

    int node_count() const { return m_mesh.node_count(); }
    int edge_count() const { return m_mesh.edge_count(); }
    double temperature(int i) const { return m_T[i]; }

    // convection per wetted edge. a coupler may recompute h from the local
    // fluid speed each step
    void set_convection(int edge, double h, double t_inf);
    void clear_convection();

    void set_fixed_temperature(int node, double T);
    void release_temperature(int node);

    void add_nodal_heat(int node, double q);
    void clear_loads() { m_q.setZero(); }

    const Mesh &mesh() const { return m_mesh; }
    const Material &material() const { return m_mat; }

  private:
    // rebuilds the robin + dirichlet lists from the per-edge convection state
    void refresh_bc();

    Mesh m_mesh;
    Material m_mat;
    std::vector<std::unique_ptr<Element>> m_elems;
    std::vector<Element *> m_elem_ptrs;

    std::unique_ptr<Assembler> m_assembler;
    BoundaryConditions m_bc;
    BackwardEuler m_integrator;

    SparseMatrix<double> m_C;  // capacity
    SparseMatrix<double> m_Kc; // conduction
    VectorXd m_T;
    VectorXd m_q;

    // per boundary edge convection state
    std::vector<char> m_conv_on;
    std::vector<double> m_conv_h;
    std::vector<double> m_conv_t_inf;

    std::vector<int> m_fixed_node;
    std::vector<double> m_fixed_T;

    // scratch stuff
    SparseMatrix<double> m_Kc_bc;
    VectorXd m_rhs;
    bool m_built = false;
};

} // namespace manifold::FEA
