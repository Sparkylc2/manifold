#pragma once

#include <manifold/fea/fea_solver.h>
#include <manifold/fluid/fluid_solver.h>

#include <cmath>
#include <vector>

namespace manifold::Coupling {
using namespace Eigen;
class FeaConvectionCoupling {
  public:
    FeaConvectionCoupling(Fluid::FluidSolver *fluid, FEA::ThermalBody *thermal,
                          double t_inf)
        : m_fluid(fluid), m_thermal(thermal), m_t_inf(t_inf) {}

    void set_base_h(double h) { m_h0 = h; }
    void set_correlation(double gain, double exponent) {
        m_gain = gain;
        m_exp = exponent;
    }
    void set_ambient(double t_inf) { m_t_inf = t_inf; }

    void set_geometry(const FEA::ElasticBody *geom) { m_geom = geom; }

    void set_edges(std::vector<int> edges) { m_edges = std::move(edges); }

    void set_conjugate(bool on) { m_conjugate = on; }

    void set_fluid_feedback(double gain) { m_feedback = gain; }

    void set_sample_offset(const Vector2d &o) { m_offset = o; }

    void update(bool velocity_correlated) {
        const FEA::Mesh &mesh = m_thermal->mesh();

        const int ne = mesh.edge_count();
        for (int idx = 0; idx < (int)active_count(ne); idx++) {
            const int i = m_edges.empty() ? idx : m_edges[idx];
            const FEA::Edge &ed = mesh.edge(i);
            const Vector2d mid =
                0.5 * (world(mesh, ed.n[0]) + world(mesh, ed.n[1])) + m_offset;

            double h = m_h0;
            if (velocity_correlated) {
                Vector2d v;
                m_fluid->velocity_at(mid, &v);
                h = film_coefficient(v.norm());
            }

            const double t_fluid =
                m_conjugate ? m_fluid->temperature_at(mid) : m_t_inf;
            m_thermal->set_convection(i, h, t_fluid);

            if (m_feedback > 0.0) {
                // energy the solid sheds across this edge, back into the fluid
                const double t_solid = 0.5 * (m_thermal->temperature(ed.n[0]) +
                                              m_thermal->temperature(ed.n[1]));
                const double len =
                    (world(mesh, ed.n[1]) - world(mesh, ed.n[0])).norm();
                const double q = h * len * (t_solid - t_fluid);
                int ci, cj;
                if (m_fluid->world_to_cell(mid, &ci, &cj))
                    m_fluid->add_heat_source(ci, cj, m_feedback * q);
            }
        }
    }

  private:
    size_t active_count(int ne) const {
        return m_edges.empty() ? (size_t)ne : m_edges.size();
    }

    double film_coefficient(double speed) const {
        return m_h0 * (1.0 + m_gain * std::pow(speed, m_exp));
    }

    Vector2d world(const FEA::Mesh &mesh, int node) const {
        return m_geom ? m_geom->node_position(node) : mesh.rest(node);
    }

    Fluid::FluidSolver *m_fluid;
    FEA::ThermalBody *m_thermal;
    const FEA::ElasticBody *m_geom = nullptr;
    std::vector<int> m_edges;
    Vector2d m_offset = Vector2d::Zero();

    double m_t_inf;
    double m_h0 = 1.0;
    double m_gain = 1.0;
    double m_exp = 0.5;
    bool m_conjugate = false;
    double m_feedback = 0.0;
};

} // namespace manifold::Coupling
