#pragma once

#include "../constraint.h"

namespace manifold::Solver {
class NoSlipConstraint : public Constraint {
  public:
    NoSlipConstraint() : Constraint(1, 1) {
        m_radius = 1.0;
        m_ks = 0.0;
        m_kd = 1.0;
        m_offset = 0.0;
        m_offset_initialized = false;
    }

    ~NoSlipConstraint() override = default;

    void calculate(Output *output, SystemState *state) override {
        const int b = m_bodies[0]->index;

        if (!m_offset_initialized) {
            m_offset = state->p[b].x() + m_radius * state->theta[b];
            m_offset_initialized = true;
        }

        output->C[0] = state->p[b].x() + m_radius * state->theta[b] - m_offset;

        // J = [1, 0, radius]
        output->J[0][0] = 1.0;
        output->J[0][1] = 0.0;
        output->J[0][2] = m_radius;

        output->J_dot[0][0] = 0.0;
        output->J_dot[0][1] = 0.0;
        output->J_dot[0][2] = 0.0;

        output->ks[0] = m_ks;
        output->kd[0] = m_kd;
        output->v_bias[0] = 0.0;

        no_limits(output);
    }

    void set_body(RigidBody *body) { m_bodies[0] = body; }
    void set_radius(double r) { m_radius = r; }
    void set_ks(double ks) { m_ks = ks; }
    void set_kd(double kd) { m_kd = kd; }

    void reset_offset() { m_offset_initialized = false; }

  private:
    double m_radius;
    double m_ks;
    double m_kd;
    double m_offset;
    bool m_offset_initialized;
};

} // namespace manifold::Solver
