#pragma once

#include "../constraint.h"
#include <cmath>

namespace manifold::Solver {

class RollingConstraint : public Constraint {
  public:
    RollingConstraint() : Constraint(2, 2) {
        m_local.setZero();
        m_direction.setZero();
        m_radius = 0.0;
        m_ks = 10.0;
        m_kd = 1.0;
    }

    ~RollingConstraint() override = default;

    void calculate(Output *output, SystemState *state) override {
        const int b0 = m_bodies[0]->index; // base
        const int b1 = m_bodies[1]->index; // roller

        const double q1 = state->p[b0].x();
        const double q2 = state->p[b0].y();
        const double q3 = state->theta[b0];

        const double q4 = state->p[b1].x();
        const double q5 = state->p[b1].y();
        const double q6 = state->theta[b1];

        const double q1_dot = state->v[b0].x();
        const double q2_dot = state->v[b0].y();
        const double q3_dot = state->v_theta[b0];

        const double q4_dot = state->v[b1].x();
        const double q5_dot = state->v[b1].y();

        const double cos_q3 = std::cos(q3);
        const double sin_q3 = std::sin(q3);

        const double lx = m_local.x();
        const double ly = m_local.y();
        const double mx = m_direction.x();
        const double my = m_direction.y();

        // surface origin in world space: origin = p0 + R0 * m_local
        const double origin_x = q1 + cos_q3 * lx - sin_q3 * ly;
        const double origin_y = q2 + sin_q3 * lx + cos_q3 * ly;

        // surface tangent in world space: d = R0 * m_direction
        const double dx = cos_q3 * mx - sin_q3 * my;
        const double dy = sin_q3 * mx + cos_q3 * my;

        // time derivatives of d (for J_dot)
        const double dx_dot = -sin_q3 * q3_dot * mx - cos_q3 * q3_dot * my;
        const double dy_dot = cos_q3 * q3_dot * mx - sin_q3 * q3_dot * my;

        // surface perpendicular (90° CCW from tangent)
        const double perp_x = -dy;
        const double perp_y = dx;

        // displacement from surface origin to roller center
        const double delta_x = q4 - origin_x;
        const double delta_y = q5 - origin_y;

        // time derivative of delta
        const double delta_x_dot =
            q4_dot - (q1_dot - sin_q3 * q3_dot * lx - cos_q3 * q3_dot * ly);
        const double delta_y_dot =
            q5_dot - (q2_dot + cos_q3 * q3_dot * lx - sin_q3 * q3_dot * ly);

        // arc-length parameter: projection of delta onto surface tangent
        const double s = delta_x * dx + delta_y * dy;

        // ---- constraint values ----

        // C0: no-slip rolling (θ_roller coupled to arc-length)
        const double C0 = -q6 - s * m_radius;

        // C1: contact (roller center stays m_radius from surface)
        const double C1 = m_radius - (perp_x * delta_x + perp_y * delta_y);

        output->C[0] = C0;
        output->C[1] = C1;

        // ---- partial derivatives for Jacobian ----
        // these are the chain rule terms for ds/dq and dC1/dq

        // d(origin)/dθ0
        const double d_origin_x_dq3 = -sin_q3 * lx - cos_q3 * ly;
        const double d_origin_y_dq3 = cos_q3 * lx - sin_q3 * ly;

        // d(delta)/d(each DOF)
        const double d_delta_x_dq1 = -1.0;
        const double d_delta_x_dq3 = -d_origin_x_dq3;
        const double d_delta_x_dq4 = 1.0;

        const double d_delta_y_dq2 = -1.0;
        const double d_delta_y_dq3 = -d_origin_y_dq3;
        const double d_delta_y_dq5 = 1.0;

        // d(d)/dθ0 = perp(d) (rotation derivative of the direction vector)
        const double d_dx_dq3 = -dy;
        const double d_dy_dq3 = dx;

        // ds/d(each base DOF): s = delta · d, product rule
        const double ds_dq1 = d_delta_x_dq1 * dx;
        const double ds_dq2 = d_delta_y_dq2 * dy;
        const double ds_dq3 = (d_delta_x_dq3 * dx + delta_x * d_dx_dq3) +
                              (d_delta_y_dq3 * dy + delta_y * d_dy_dq3);

        // ---- J for C0: dC0/dq = [-ds/dq * radius, ..., 0, 0, -1] ----

        // body 0 (base)
        output->J[0][0] = -ds_dq1 * m_radius;
        output->J[0][1] = -ds_dq2 * m_radius;
        output->J[0][2] = -ds_dq3 * m_radius;

        // body 1 (roller): ds/dp1 = d, ds/dθ1 = 0, dC0/dθ1 = -1
        output->J[0][3] = -dx * m_radius;
        output->J[0][4] = -dy * m_radius;
        output->J[0][5] = -1.0;

        // ---- J for C1: dC1/dq = -d(perp · delta)/dq ----

        // body 0
        output->J[1][0] = dy * d_delta_x_dq1;
        output->J[1][1] = -dx * d_delta_y_dq2;
        output->J[1][2] = (d_dy_dq3 * delta_x + dy * d_delta_x_dq3) -
                          (d_dx_dq3 * delta_y + dx * d_delta_y_dq3);

        // body 1
        output->J[1][3] = dy * d_delta_x_dq4;
        output->J[1][4] = -dx * d_delta_y_dq5;
        output->J[1][5] = 0.0;

        // ---- J_dot (time derivatives of J entries) ----
        // needed for the Baumgarte RHS: J_dot * q_dot term

        const double d_dx_dq3_dot =
            -cos_q3 * q3_dot * mx + sin_q3 * q3_dot * my;
        const double d_dy_dq3_dot =
            -sin_q3 * q3_dot * mx - cos_q3 * q3_dot * my;
        const double d_delta_x_dq3_dot =
            cos_q3 * q3_dot * lx - sin_q3 * q3_dot * ly;
        const double d_delta_y_dq3_dot =
            sin_q3 * q3_dot * lx + cos_q3 * q3_dot * ly;

        const double ds_dq1_dot = d_delta_x_dq1 * dx_dot;
        const double ds_dq2_dot = d_delta_y_dq2 * dy_dot;
        const double ds_dq3_dot =
            (d_delta_x_dq3_dot * dx + d_delta_x_dq3 * dx_dot) +
            (delta_x_dot * d_dx_dq3 + delta_x * d_dx_dq3_dot) +
            (d_delta_y_dq3_dot * dy + d_delta_y_dq3 * dy_dot) +
            (delta_y_dot * d_dy_dq3 + delta_y * d_dy_dq3_dot);

        // J_dot for C0
        output->J_dot[0][0] = -ds_dq1_dot * m_radius;
        output->J_dot[0][1] = -ds_dq2_dot * m_radius;
        output->J_dot[0][2] = -ds_dq3_dot * m_radius;
        output->J_dot[0][3] = -dx_dot * m_radius;
        output->J_dot[0][4] = -dy_dot * m_radius;
        output->J_dot[0][5] = 0.0;

        // J_dot for C1
        output->J_dot[1][0] = dy_dot * d_delta_x_dq1;
        output->J_dot[1][1] = -dx_dot * d_delta_y_dq2;
        output->J_dot[1][2] =
            (d_dy_dq3_dot * delta_x + d_dy_dq3 * delta_x_dot) +
            (dy_dot * d_delta_x_dq3 + dy * d_delta_x_dq3_dot) -
            (d_dx_dq3_dot * delta_y + d_dx_dq3 * delta_y_dot) -
            (dx_dot * d_delta_y_dq3 + dx * d_delta_y_dq3_dot);
        output->J_dot[1][3] = dy_dot * d_delta_x_dq4;
        output->J_dot[1][4] = -dx_dot * d_delta_y_dq5;
        output->J_dot[1][5] = 0.0;

        // ---- stabilization ----

        // no position correction on no-slip (causes jitter)
        output->ks[0] = 0.0;
        output->kd[0] = 0.0;

        // position correction on contact distance
        output->ks[1] = m_ks;
        output->kd[1] = m_kd;

        output->v_bias[0] = 0.0;
        output->v_bias[1] = 0.0;

        no_limits(output);
    }

    void set_base_body(RigidBody *body) { m_bodies[0] = body; }
    void set_rolling_body(RigidBody *body) { m_bodies[1] = body; }

    void set_local_origin(const Vector2d &origin) { m_local = origin; }
    void set_direction(const Vector2d &dir) { m_direction = dir; }
    void set_radius(double r) { m_radius = r; }

    void set_ks(double ks) { m_ks = ks; }
    void set_kd(double kd) { m_kd = kd; }

  private:
    Vector2d m_local;
    Vector2d m_direction;
    double m_radius;
    double m_ks;
    double m_kd;
};

} // namespace manifold::Solver
