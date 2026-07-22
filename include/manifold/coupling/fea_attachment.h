#pragma once

#include <manifold/fea/fea_solver.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Coupling {
using namespace Eigen;

// hard attachment: an FEA node is pinned to a point on a rigid body (e.g. a cart
// on a track). each tick the node's Dirichlet target is retargeted to the body
// point's world position, so cart motion drags the node and stress develops in
// the connection. use instead of FeaSpringForce when the joint should be rigid.
//
// note: this is one-way (body -> node). the node exerts no reaction back on the
// body here; add a reaction through the force path if the body should feel it.
class FeaCartAttachment {
  public:
    FeaCartAttachment(const Solver::RigidBody *body, const Vector2d &local,
                      FEA::FeaSolver *fea, int node)
        : m_body(body), m_local(local), m_fea(fea), m_node(node) {}

    // retarget the node's Dirichlet BC to the current body point. call once per
    // tick before fea->advance().
    void update() {
        Vector2d w;
        m_body->local_to_world(m_local, &w);
        m_fea->set_fixed_node(m_node, w);
    }

  private:
    const Solver::RigidBody *m_body;
    Vector2d m_local;
    FEA::FeaSolver *m_fea;
    int m_node;
};

} // namespace manifold::Coupling
