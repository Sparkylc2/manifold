#pragma once

#include <manifold/fea/fea_solver.h>
#include <manifold/fluid/solid_boundary.h>

#include <algorithm>
#include <vector>

namespace manifold::Coupling {
using namespace Eigen;

// presents a deformed FEA body to the fluid as a SolidBoundary.
//
// the shape is a capsule chain: the caller supplies the node indices along the
// body's centreline plus a half thickness, and the sdf is the distance to that
// polyline minus the radius. bending falls out for free
//
// this is the two-way path. the fluid re-reads boundaries every advance, so the
// solid it sees genuinely moves with the structure. weakly coupled though, so a
// light structure in a dense fluid can hit added-mass instability
class FeaBodyBoundary : public Fluid::SolidBoundary {
  public:
    FeaBodyBoundary(const FEA::FeaSolver *body, std::vector<int> chain,
                    double half_thickness)
        : m_body(body), m_chain(std::move(chain)), m_r(half_thickness) {
        refresh();
    }

    // the fluid evaluates the sdf per grid cell, so snapshot the chain once a
    // tick instead of calling back into the solver for every cell
    void refresh() {
        m_p.resize(m_chain.size());
        m_v.resize(m_chain.size());
        for (size_t i = 0; i < m_chain.size(); i++) {
            m_p[i] = m_body->node_position(m_chain[i]);
            m_v[i] = m_body->node_velocity(m_chain[i]);
        }
    }

    double signed_distance(const Vector2d &x) const override {
        double best = 1e30;
        for (size_t i = 0; i + 1 < m_p.size(); i++) {
            double t;
            best = std::min(best, seg_distance(x, m_p[i], m_p[i + 1], &t));
        }
        return best - m_r;
    }

    void velocity_at(const Vector2d &x, Vector2d *v) const override {
        double best = 1e30;
        double best_t = 0.0;
        size_t best_i = 0;

        for (size_t i = 0; i + 1 < m_p.size(); i++) {
            double t;
            const double d = seg_distance(x, m_p[i], m_p[i + 1], &t);
            if (d < best) {
                best = d;
                best_t = t;
                best_i = i;
            }
        }

        if (m_v.size() < 2) {
            *v = Vector2d::Zero();
            return;
        }
        *v = (1.0 - best_t) * m_v[best_i] + best_t * m_v[best_i + 1];
    }

    void set_half_thickness(double r) { m_r = r; }

  private:
    static double seg_distance(const Vector2d &x, const Vector2d &a,
                               const Vector2d &b, double *t_out) {
        const Vector2d ab = b - a;
        const double len2 = ab.squaredNorm();
        double t = 0.0;
        if (len2 > 1e-18)
            t = std::clamp((x - a).dot(ab) / len2, 0.0, 1.0);
        *t_out = t;
        return (x - (a + t * ab)).norm();
    }

    const FEA::FeaSolver *m_body;
    std::vector<int> m_chain;
    double m_r;

    std::vector<Vector2d> m_p;
    std::vector<Vector2d> m_v;
};

} // namespace manifold::Coupling
