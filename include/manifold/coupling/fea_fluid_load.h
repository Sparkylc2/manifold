#pragma once

#include <manifold/fea/fea_solver.h>
#include <manifold/fluid/fluid_solver.h>

#include <vector>

namespace manifold::Coupling {
using namespace Eigen;

// fluid surface tractions -> FEA nodal loads.
// walks the mesh boundary edges, samples the fluid pressure just outside each
// wetted edge and integrates the traction onto its two nodes
//      t = -p*n,  f_i = integral of N_i*t dS = t*L/2 for a linear edge
//
// under-relaxation: staggered two-way coupling with a hard fluid BC hits the
// added-mass instability, the fluid over-reacts to the boundary motion and the
// feedback diverges. blending the load with the previous tick's,
//      f = w*f_new + (1-w)*f_prev
// damps that. w=1 is off (fine for one-way / rigid), w~0.3 stabilises two-way
class FeaFluidLoad {
  public:
    FeaFluidLoad(const Fluid::FluidSolver *fluid, FEA::ElasticBody *body)
        : m_fluid(fluid), m_body(body) {}

    // sample point is pushed this far along the outward normal, so it lands in
    // fluid rather than inside the solid
    void set_sample_offset(double d) { m_offset = d; }
    void set_scale(double s) { m_scale = s; }
    void set_relax(double w) { m_relax = w; }

    void update() {
        const FEA::Mesh &mesh = m_body->mesh();
        const int nn = mesh.node_count();
        if ((int)m_load.size() != nn) {
            m_load.assign(nn, Vector2d::Zero());
            m_prev.assign(nn, Vector2d::Zero());
        }

        // accumulate this tick's raw nodal load
        for (auto &f : m_load)
            f.setZero();
        for (int i = 0; i < mesh.edge_count(); i++) {
            const FEA::Edge &ed = mesh.edge(i);
            const Vector2d a = m_body->node_position(ed.n[0]);
            const Vector2d b = m_body->node_position(ed.n[1]);

            const Vector2d d = b - a;
            const double len = d.norm();
            if (len <= 1e-12)
                continue;

            // ccw winding, so the outward normal is d rotated -90
            const Vector2d n(d.y() / len, -d.x() / len);
            const Vector2d mid = 0.5 * (a + b) + m_offset * n;

            const double p = m_fluid->pressure_at(mid);
            const Vector2d half = (-p * m_scale * 0.5 * len) * n;
            m_load[ed.n[0]] += half;
            m_load[ed.n[1]] += half;
        }

        // blend with last tick, then apply
        for (int i = 0; i < nn; i++) {
            const Vector2d f = m_relax * m_load[i] + (1.0 - m_relax) * m_prev[i];
            m_body->add_nodal_force(i, f);
            m_prev[i] = f;
        }
    }

  private:
    const Fluid::FluidSolver *m_fluid;
    FEA::ElasticBody *m_body;
    double m_offset = 0.0;
    double m_scale = 1.0;
    double m_relax = 1.0;

    std::vector<Vector2d> m_load;
    std::vector<Vector2d> m_prev;
};

} // namespace manifold::Coupling
