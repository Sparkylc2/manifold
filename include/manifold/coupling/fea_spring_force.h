#pragma once

#include <manifold/fea/fea_solver.h>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Coupling {
using namespace Eigen;

// two-sided spring tying an FEA node either to a world-fixed anchor or to a
// point on a rigid body:
//      F = -k*(|d| - L)*dhat - c*(v_node - v_attach),   d = x_node - x_attach
// +F onto the FEA node, -F onto the rigid body (if there is one).
// evaluated once per coupled tick and held constant across the solver
// substeps, same as FluidWrenchForce
class FeaSpringForce : public Solver::ForceGenerator {
  public:
    FeaSpringForce() = default;

    FeaSpringForce(const Vector2d &global_attach, FEA::FeaSolver *fea, int node,
                   double k, double c)
        : m_global(global_attach), m_fea(fea), m_node(node), m_k(k), m_c(c) {}

    FeaSpringForce(const Solver::RigidBody *body, const Vector2d &local_attach,
                   FEA::FeaSolver *fea, int node, double k, double c)
        : m_body(body), m_local(local_attach), m_fea(fea), m_node(node), m_k(k),
          m_c(c) {}

    void update() {
        if (m_fea == nullptr)
            return;

        Vector2d attach = m_global;
        Vector2d v_attach = Vector2d::Zero();

        if (m_body != nullptr) {
            m_body->local_to_world(m_local, &attach);
            // v + omega x r at the attach point
            const Vector2d r = attach - m_body->p;
            const double w = m_body->v_theta;
            v_attach = m_body->v + Vector2d(-w * r.y(), w * r.x());
        }

        const Vector2d d = m_fea->node_position(m_node) - attach;
        const Vector2d dv = m_fea->node_velocity(m_node) - v_attach;

        Vector2d f = -m_c * dv;
        if (m_rest > 0.0) {
            // rest length needs a direction; the linear form below is the
            // L = 0 limit and stays well-defined as |d| -> 0
            const double len = d.norm();
            if (len > 1e-9)
                f -= m_k * (len - m_rest) * (d / len);
        } else {
            f -= m_k * d;
        }

        m_fea->add_nodal_force(m_node, f);
        m_react = -f;
        m_attach = attach;
    }

    void apply(Solver::SystemState *state) override {
        if (m_body == nullptr)
            return;

        const int i = m_body->index;
        if (i < 0)
            return;
        state->apply_force(m_local, m_react, i);
    }

    void set_stiffness(double k) { m_k = k; }
    void set_damping(double c) { m_c = c; }
    void set_rest_length(double l) { m_rest = l; }

    int node() const { return m_node; }
    const Vector2d &attach_point() const { return m_attach; }

  private:
    const Solver::RigidBody *m_body = nullptr;
    Vector2d m_local = Vector2d::Zero();
    Vector2d m_global = Vector2d::Zero();
    FEA::FeaSolver *m_fea = nullptr;
    int m_node = 0;
    double m_k = 0.0;
    double m_c = 0.0;
    double m_rest = 0.0;

    Vector2d m_attach = Vector2d::Zero();
    Vector2d m_react = Vector2d::Zero();
};

} // namespace manifold::Coupling
