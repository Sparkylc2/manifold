#pragma once

#include "../constraint.h"
#include <cfloat>

namespace manifold::Solver {

class GearConstraint : public Constraint {
  public:
    GearConstraint() : Constraint(1, 2) {
        m_ratio = 1.0;
        m_ks = 10.0;
        m_kd = 1.0;
        m_neutral = false;
        m_correct_position = true;
        m_offset = 0.0;
        m_offset_initialized = false;
    }

    ~GearConstraint() override = default;

    void calculate(Output *output, SystemState *state) override {
        const int b0 = m_bodies[0]->index;
        const int b1 = m_bodies[1]->index;

        if (!m_offset_initialized) {
            m_offset = state->theta[b0] - m_ratio * state->theta[b1];
            m_offset_initialized = true;
        }

        if (m_correct_position) {
            output->C[0] =
                state->theta[b0] - m_ratio * state->theta[b1] - m_offset;
        } else {
            output->C[0] = 0.0;
        }

        // J: only theta DOFs are non-zero
        // body 0: [0, 0, 1]
        output->J[0][0] = 0.0;
        output->J[0][1] = 0.0;
        output->J[0][2] = 1.0;

        // body 1: [0, 0, -ratio]
        output->J[0][3] = 0.0;
        output->J[0][4] = 0.0;
        output->J[0][5] = -m_ratio;

        // J_dot: all constant, so zero
        output->J_dot[0][0] = 0.0;
        output->J_dot[0][1] = 0.0;
        output->J_dot[0][2] = 0.0;
        output->J_dot[0][3] = 0.0;
        output->J_dot[0][4] = 0.0;
        output->J_dot[0][5] = 0.0;

        output->ks[0] = m_ks;
        output->kd[0] = m_kd;
        output->v_bias[0] = 0.0;

        if (!m_neutral) {
            no_limits(output);
        } else {
            output->limits[0][0] = 0.0;
            output->limits[0][1] = 0.0;
        }
    }

    void set_bodies(RigidBody *b0, RigidBody *b1) {
        m_bodies[0] = b0;
        m_bodies[1] = b1;
    }

    void set_ratio(double ratio) { m_ratio = ratio; }
    double ratio() const { return m_ratio; }

    void set_neutral(bool n) { m_neutral = n; }
    bool neutral() const { return m_neutral; }

    void set_correct_position(bool c) { m_correct_position = c; }
    void set_ks(double ks) { m_ks = ks; }
    void set_kd(double kd) { m_kd = kd; }

    void reset_offset() { m_offset_initialized = false; }

  private:
    double m_ratio;
    double m_ks;
    double m_kd;
    bool m_neutral;
    bool m_correct_position;
    double m_offset;
    bool m_offset_initialized;
};

} // namespace manifold::Solver
