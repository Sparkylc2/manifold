#pragma once

#include <manifold/fea/fea_solver.h>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/rigid_body.h>

namespace manifold::Coupling {
using namespace Eigen;

// two-sided spring tying an FEA attachment either to a world-fixed anchor or
// to a point on a rigid body:
//      F = -k*(|d| - L)*dhat - c*(v_att - v_attach),   d = x_att - x_attach
// +F onto the FEA attachment, -F onto the rigid body (if there is one).
// evaluated once per coupled tick and held constant across the solver
// substeps, same as FluidWrenchForce
//
// the attachment is a weighted set of nodes, not necessarily one. k and c are
// explicit, so a single node caps the step at dt < 2*m_node/c -- and m_node
// falls off like 1/(nx*ny), so a point mount silently detonates the moment the
// mesh is refined. spreading the same k, c over a footprint of fixed physical
// size holds m constant under refinement, and is what a bolted mount is anyway
class FeaSpringForce : public Solver::ForceGenerator {
  public:
    FeaSpringForce() = default;

    FeaSpringForce(const Vector2d &global_attach, FEA::FeaSolver *fea, int node,
                   double k, double c)
        : m_global(global_attach), m_fea(fea), m_nodes{node}, m_w{1.0}, m_k(k),
          m_c(c) {}

    FeaSpringForce(const Solver::RigidBody *body, const Vector2d &local_attach,
                   FEA::FeaSolver *fea, int node, double k, double c)
        : m_body(body), m_local(local_attach), m_fea(fea), m_nodes{node},
          m_w{1.0}, m_k(k), m_c(c) {}

    // Constrain the spring to act along one direction, leaving the attachment
    // free to slide along the perpendicular -- a roller on a rail rather than a
    // pin. Both the stiffness and the damping are projected, so the free axis
    // carries no force at all. Pass a zero vector to go back to acting in both.
    void set_axis(const Vector2d &dir) {
        const double n = dir.norm();
        m_axis = n > 1e-9 ? Vector2d(dir / n) : Vector2d::Zero();
    }

    // Viscous friction on the free axis of a railed mount. It resists sliding
    // without pulling the mount toward any particular place along the rail, so
    // the attachment still travels -- it just cannot run away. Without it a
    // steady cross-load has nothing at all to work against and the mount
    // accelerates off the rail. No effect unless set_axis() is in use.
    void set_rail_friction(double c) { m_slide_c = c; }

    // weights are normalised to sum to 1; pass each node's share of the
    // attachment mass so every node in the patch sees the same k/m
    void set_footprint(std::vector<int> nodes, std::vector<double> weights) {
        if (nodes.empty() || nodes.size() != weights.size())
            return;
        double sum = 0.0;
        for (double w : weights)
            sum += w;
        if (sum <= 0.0)
            return;
        for (double &w : weights)
            w /= sum;
        m_nodes = std::move(nodes);
        m_w = std::move(weights);
    }

    void update() {
        if (m_fea == nullptr || m_nodes.empty())
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

        // the patch acts as one attachment point, so the spring sees its
        // pooled mass rather than whatever a single node happens to carry
        Vector2d x_att = Vector2d::Zero(), v_att = Vector2d::Zero();
        for (size_t i = 0; i < m_nodes.size(); i++) {
            x_att += m_w[i] * m_fea->node_position(m_nodes[i]);
            v_att += m_w[i] * m_fea->node_velocity(m_nodes[i]);
        }

        Vector2d d = x_att - attach;
        Vector2d dv = v_att - v_attach;

        // on a rail, only the along-axis part of the offset and of the closing
        // velocity is resisted -- what the mount does perpendicular to it is
        // free, which is what makes the rest length below a distance measured
        // along the rail rather than a radius about the anchor
        Vector2d f_slide = Vector2d::Zero();
        if (m_axis.squaredNorm() > 0.0) {
            const Vector2d d_ax = m_axis * m_axis.dot(d);
            const Vector2d dv_ax = m_axis * m_axis.dot(dv);
            f_slide = -m_slide_c * (dv - dv_ax);
            d = d_ax;
            dv = dv_ax;
        }

        Vector2d f = -m_c * dv + f_slide;
        if (m_rest > 0.0) {
            // rest length needs a direction; the linear form below is the
            // L = 0 limit and stays well-defined as |d| -> 0
            const double len = d.norm();
            if (len > 1e-9)
                f -= m_k * (len - m_rest) * (d / len);
        } else {
            f -= m_k * d;
        }

        for (size_t i = 0; i < m_nodes.size(); i++)
            m_fea->add_nodal_force(m_nodes[i], m_w[i] * f);
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

    int node() const { return m_nodes.empty() ? 0 : m_nodes.front(); }
    const std::vector<int> &nodes() const { return m_nodes; }
    const Vector2d &attach_point() const { return m_attach; }

  private:
    const Solver::RigidBody *m_body = nullptr;
    Vector2d m_local = Vector2d::Zero();
    Vector2d m_global = Vector2d::Zero();
    FEA::FeaSolver *m_fea = nullptr;
    std::vector<int> m_nodes;
    std::vector<double> m_w;
    Vector2d m_axis = Vector2d::Zero(); // zero -> acts in both axes
    double m_slide_c = 0.0;             // friction on the free axis
    double m_k = 0.0;
    double m_c = 0.0;
    double m_rest = 0.0;

    Vector2d m_attach = Vector2d::Zero();
    Vector2d m_react = Vector2d::Zero();
};

} // namespace manifold::Coupling
