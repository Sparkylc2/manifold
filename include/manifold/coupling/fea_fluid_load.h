#pragma once

#include <manifold/fea/fea_solver.h>
#include <manifold/fluid/fluid_solver.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Coupling {
using namespace Eigen;

// integrates fluid pressure over the FEA surface into nodal loads

class FeaFluidLoad {
  public:
    FeaFluidLoad(const Fluid::FluidSolver *fluid, FEA::ElasticBody *body)
        : m_fluid(fluid), m_body(body) {}

    // sample point is pushed this far along the outward normal, so it lands in
    // fluid rather than inside the solid
    void set_sample_offset(double d) { m_offset = d; }
    void set_scale(double s) { m_scale = s; }

    // fixed relaxation, disables Aitken
    void set_relax(double w) {
        m_relax = w;
        m_aitken = false;
    }

    void set_aitken(double w0 = 0.05, double w_min = 0.002,
                    double w_max = 0.5) {
        m_aitken = true;
        m_omega = w0;
        m_omega_min = w_min;
        m_omega_max = w_max;
    }

    double omega() const { return m_aitken ? m_omega : m_relax; }

    void update() {
        const FEA::Mesh &mesh = m_body->mesh();
        const int nn = mesh.node_count();
        if ((int)m_load.size() != nn) {
            m_load.assign(nn, Vector2d::Zero());
            m_prev.assign(nn, Vector2d::Zero());
            m_res.assign(nn, Vector2d::Zero());
            m_res_prev.assign(nn, Vector2d::Zero());
            m_primed = false;
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

        for (int i = 0; i < nn; i++)
            m_res[i] = m_load[i] - m_prev[i];

        const double w = m_aitken ? aitken_omega(nn) : m_relax;

        for (int i = 0; i < nn; i++) {
            const Vector2d f = m_prev[i] + w * m_res[i];
            m_body->add_nodal_force(i, f);
            m_prev[i] = f;
            m_res_prev[i] = m_res[i];
        }
        m_primed = true;
    }

    // re-adds the load update() last applied, without resampling the fluid.
    // the structure may need several steps per fluid step to stay inside its
    // own stability limit, and each one clears the load accumulator first
    void reapply() {
        for (size_t i = 0; i < m_prev.size(); i++)
            m_body->add_nodal_force((int)i, m_prev[i]);
    }

  private:
    // omega_k = -omega_{k-1} * <r_{k-1}, dr> / <dr, dr>,  dr = r_k - r_{k-1}
    double aitken_omega(int nn) {
        if (!m_primed)
            return m_omega;

        double num = 0.0, den = 0.0;
        for (int i = 0; i < nn; i++) {
            const Vector2d dr = m_res[i] - m_res_prev[i];
            num += m_res_prev[i].dot(dr);
            den += dr.squaredNorm();
        }
        if (den > 1e-30) {
            const double w = -m_omega * num / den;
            if (std::isfinite(w))
                m_omega = std::clamp(w, m_omega_min, m_omega_max);
        }
        return m_omega;
    }

    const Fluid::FluidSolver *m_fluid;
    FEA::ElasticBody *m_body;
    double m_offset = 0.0;
    double m_scale = 1.0;
    double m_relax = 1.0;

    bool m_aitken = false;
    double m_omega = 0.05, m_omega_min = 0.002, m_omega_max = 0.5;
    bool m_primed = false;

    std::vector<Vector2d> m_load, m_prev, m_res, m_res_prev;
};

// TODO: true partitioned FSI subiterates *within* a step
// restore fluid and structure to t_n, re-solve both against the relaxed load,
// repeat until the interface residual converges that needs checkpoint/restore
// on StableFluidSolver (its Field2Ds are plain vectors, so this is a memcpy)
// and on ElasticBody's DOF vector worth having before the rocket's thrust comes
// from its own exhaust — the same free-body configuration that breaks the
// single-pass scheme here.

} // namespace manifold::Coupling
