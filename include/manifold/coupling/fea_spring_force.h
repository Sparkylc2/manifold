#pragma once

#include <manifold/fea/fea_solver.h>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Coupling {
using namespace Eigen;

// two-sided spring tying a point on a rigid body to an FEA node
//      F = -k*(x_node - x_attach) - c*(v_node - v_attach)
// +F onto the FEA node, -F onto the rigid body.
// evaluated once per coupled tick and held constant across the solver
// substeps, same as FluidWrenchForce
class FeaSpringForce : public Solver::ForceGenerator {
  public:
    FeaSpringForce(const Solver::RigidBody *body, const Vector2d &local_attach,
                   FEA::FeaSolver *fea, int node, double k, double c)
        : m_body(body), m_local(local_attach), m_fea(fea), m_node(node), m_k(k),
          m_c(c) {}

    void update() {
        Vector2d attach;
        m_body->local_to_world(m_local, &attach);

        // v + omega x r at the attach point
        const Vector2d r = attach - m_body->p;
        const double w = m_body->v_theta;
        const Vector2d v_attach = m_body->v + Vector2d(-w * r.y(), w * r.x());

        const Vector2d dx = m_fea->node_position(m_node) - attach;
        const Vector2d dv = m_fea->node_velocity(m_node) - v_attach;

        const Vector2d f = -m_k * dx - m_c * dv;
        m_fea->add_nodal_force(m_node, f);
        m_react = -f;
    }

    void apply(Solver::SystemState *state) override {
        const int i = m_body->index;
        if (i < 0)
            return;
        state->apply_force(m_local, m_react, i);
    }

    void set_stiffness(double k) { m_k = k; }
    void set_damping(double c) { m_c = c; }

  private:
    const Solver::RigidBody *m_body;
    Vector2d m_local;
    FEA::FeaSolver *m_fea;
    int m_node;
    double m_k;
    double m_c;

    Vector2d m_react = Vector2d::Zero();
};

} // namespace manifold::Coupling
