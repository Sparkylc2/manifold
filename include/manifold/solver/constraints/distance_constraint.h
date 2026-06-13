#pragma once

#include "../constraint.h"
#include <cmath>

namespace manifold::Solver {

class DistanceConstraint : public Constraint {
  public:
    DistanceConstraint() : Constraint(1, 2) {
        m_distance = 1.0;
        m_ks = 10.0;
        m_kd = 1.0;
    }

    ~DistanceConstraint() override = default;

    void calculate(Output *output, SystemState *state) override {
        const int b0 = m_bodies[0]->index;
        const int b1 = m_bodies[1]->index;

        Vector2d delta = state->p[b0] - state->p[b1];
        double len = delta.norm();

        if (len < 1e-10) {
            // degenerate — pick arbitrary direction
            delta = Vector2d(1, 0);
            len = 1e-10;
        }

        Vector2d n = delta / len;

        // C = |p0 - p1| - L
        output->C[0] = len - m_distance;

        // J = [n.x, n.y, 0, -n.x, -n.y, 0]
        output->J[0][0] = n.x();
        output->J[0][1] = n.y();
        output->J[0][2] = 0;
        output->J[0][3] = -n.x();
        output->J[0][4] = -n.y();
        output->J[0][5] = 0;

        // J_dot: derivative of n w.r.t. time
        Vector2d v_rel = state->v[b0] - state->v[b1];
        double v_along = v_rel.dot(n);
        Vector2d n_dot = (v_rel - v_along * n) / len;

        output->J_dot[0][0] = n_dot.x();
        output->J_dot[0][1] = n_dot.y();
        output->J_dot[0][2] = 0;
        output->J_dot[0][3] = -n_dot.x();
        output->J_dot[0][4] = -n_dot.y();
        output->J_dot[0][5] = 0;

        output->ks[0] = m_ks;
        output->kd[0] = m_kd;
        output->v_bias[0] = 0;
        no_limits(output);
    }

    void set_bodies(RigidBody *a, RigidBody *b) {
        m_bodies[0] = a;
        m_bodies[1] = b;
    }

    void set_distance(double d) { m_distance = d; }

    // automatically compute from current body positions
    void set_distance_from_bodies() {
        if (m_bodies[0] && m_bodies[1])
            m_distance = (m_bodies[0]->p - m_bodies[1]->p).norm();
    }

    double distance() const { return m_distance; }

    void set_ks(double ks) { m_ks = ks; }
    void set_kd(double kd) { m_kd = kd; }

  private:
    double m_distance;
    double m_ks;
    double m_kd;
};

} // namespace manifold::Solver
