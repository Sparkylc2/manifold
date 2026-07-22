#pragma once

#include <manifold/fea/fea_solver.h>
#include <manifold/fluid/fluid_solver.h>

#include <cmath>

namespace manifold::Coupling {
using namespace Eigen;

// convective exchange between a ThermalBody's wetted surface and the fluid.
// each tick it sets the robin BC (h, t_inf) on every boundary edge
//
// tier 1: h is prescribed and uniform
// tier 2: h tracks the local fluid speed through a nusselt-style correlation
//         h = h0*(1 + gain*speed^n), reusing FluidSolver::velocity_at
//
// tier 3 (fluid advects its own temperature, true conjugate transfer) is out
// of scope, the FluidSolver interface carries no temperature field
class FeaConvectionCoupling {
  public:
    FeaConvectionCoupling(const Fluid::FluidSolver *fluid,
                          FEA::ThermalBody *thermal, double t_inf)
        : m_fluid(fluid), m_thermal(thermal), m_t_inf(t_inf) {}

    void set_base_h(double h) { m_h0 = h; }
    void set_correlation(double gain, double exponent) {
        m_gain = gain;
        m_exp = exponent;
    }
    void set_ambient(double t_inf) { m_t_inf = t_inf; }

    // the thermal mesh only stores rest coords, so if the body is also being
    // deformed/carried around, point this at the elastic body to sample the
    // fluid where the surface actually is
    void set_geometry(const FEA::ElasticBody *geom) { m_geom = geom; }

    void update(bool velocity_correlated) {
        const FEA::Mesh &mesh = m_thermal->mesh();

        for (int i = 0; i < mesh.edge_count(); i++) {
            double h = m_h0;

            if (velocity_correlated) {
                const FEA::Edge &ed = mesh.edge(i);
                const Vector2d mid = 0.5 * (world(mesh, ed.n[0]) + world(mesh, ed.n[1]));

                Vector2d v;
                m_fluid->velocity_at(mid, &v);
                h = film_coefficient(v.norm());
            }

            m_thermal->set_convection(i, h, m_t_inf);
        }
    }

  private:
    double film_coefficient(double speed) const {
        return m_h0 * (1.0 + m_gain * std::pow(speed, m_exp));
    }

    Vector2d world(const FEA::Mesh &mesh, int node) const {
        return m_geom ? m_geom->node_position(node) : mesh.rest(node);
    }

    const Fluid::FluidSolver *m_fluid;
    FEA::ThermalBody *m_thermal;
    const FEA::ElasticBody *m_geom = nullptr;

    double m_t_inf;
    double m_h0 = 1.0;
    double m_gain = 1.0;
    double m_exp = 0.5;
};

} // namespace manifold::Coupling
