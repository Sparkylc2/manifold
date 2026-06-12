#pragma once
#include <cassert>
#include <cfloat>
#include <manifold/solver/rigid_body.h>
#include <manifold/solver/system_state.h>

namespace manifold::Solver {

class Constraint {
  public:
    static constexpr int MAX_CONSTRAINT_COUNT = 3;
    static constexpr int MAX_BODY_COUNT = 2;

    struct Output {
        double C[MAX_CONSTRAINT_COUNT];
        double J[MAX_CONSTRAINT_COUNT][3 * MAX_BODY_COUNT];
        double J_dot[MAX_CONSTRAINT_COUNT][3 * MAX_BODY_COUNT];
        double v_bias[MAX_CONSTRAINT_COUNT];
        double limits[MAX_CONSTRAINT_COUNT][2];
        double ks[MAX_CONSTRAINT_COUNT];
        double kd[MAX_CONSTRAINT_COUNT];
    };

    Constraint(int constraint_count, int body_count);
    virtual ~Constraint() = default;

    virtual void calculate(Output *output, SystemState *state) = 0;
    int constraint_count() const { return m_constraint_count; }

    int m_index = -1;
    int m_body_count;
    RigidBody *m_bodies[MAX_BODY_COUNT] = {};

    Vector2d F_xy[MAX_CONSTRAINT_COUNT][MAX_BODY_COUNT];
    double F_t[MAX_CONSTRAINT_COUNT][MAX_BODY_COUNT];

    void set_visible(bool v) { m_visible = v; }
    bool visible() const { return m_visible; }

  protected:
    bool m_visible = true;
    int m_constraint_count;

    static void no_limits(Output *output) {
        for (int i = 0; i < MAX_CONSTRAINT_COUNT; ++i) {
            output->limits[i][0] = -DBL_MAX;
            output->limits[i][1] = DBL_MAX;
        }
    }
};

} // namespace manifold::Solver
