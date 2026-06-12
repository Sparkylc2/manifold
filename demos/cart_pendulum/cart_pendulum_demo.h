#pragma once
#include <manifold/control/pid.h>
#include <manifold/renderer/renderer.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/direct_force.h>
#include <manifold/solver/forces/gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <raylib.h>

#include <manifold/solver/rk4_ode_solver.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
class CartPendulumDemo {
  public:
    static constexpr double CartMass = 5.0;
    static constexpr double CartWidth = 1.2;
    static constexpr double CartHeight = 0.5;
    static constexpr double PendulumMass = 1.0;
    static constexpr double PendulumLen = 3.0;
    static constexpr double PendulumWidth = 0.12;
    static constexpr double Gravity = 9.81;
    static constexpr int SimSteps = 100;
    static constexpr int MaxHistory = 600;

    // tuner state
    enum class TunerTarget { Angle, Position };
    enum class TunerParam { Kp, Ki, Kd };

    void initialize() {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_cart.reset();
        m_cart.m = CartMass;
        m_cart.I =
            CartMass * (CartWidth * CartWidth + CartHeight * CartHeight) / 12.0;
        m_cart.p = Vector2d(0, 0);
        m_system.add_body(&m_cart);

        double angle = M_PI / 2.0 + 0.1;
        m_pendulum.reset();
        m_pendulum.m = PendulumMass;
        m_pendulum.I = PendulumMass * PendulumLen * PendulumLen / 12.0;
        m_pendulum.theta = angle;
        m_pendulum.p = Vector2d(std::cos(angle) * PendulumLen / 2.0,
                                std::sin(angle) * PendulumLen / 2.0);
        m_system.add_body(&m_pendulum);

        m_rail.set_body(&m_cart);
        m_rail.set_line(Vector2d(0, 0), Vector2d(1, 0));
        m_rail.set_local_pos(Vector2d(0, 0));
        m_rail.set_ks(100.0);
        m_rail.set_kd(10.0);
        m_system.add_constraint(&m_rail);

        m_cart_rot.set_body(&m_cart);
        m_cart_rot.set_angle(0);
        m_cart_rot.set_ks(100.0);
        m_cart_rot.set_kd(10.0);
        m_system.add_constraint(&m_cart_rot);

        m_pivot.set_bodies(&m_cart, &m_pendulum);
        m_pivot.set_local_pos1(Vector2d(0, 0));
        m_pivot.set_local_pos2(Vector2d(-PendulumLen / 2.0, 0));
        m_pivot.set_ks(100.0);
        m_pivot.set_kd(10.0);
        m_system.add_constraint(&m_pivot);

        m_gravity.set_gravity(Gravity);
        m_system.add_force_generator(&m_gravity);

        m_control_force.set_body(&m_cart);
        m_system.add_force_generator(&m_control_force);

        apply_gains();

        m_angle_hist.clear();
        m_cart_hist.clear();
        m_force_hist.clear();
        m_energy_hist.clear();
        m_time = 0;
        m_last_force = 0;
    }

    void process(double dt) {
        double angle_err = angle_error(m_pendulum.theta, M_PI / 2.0);
        double pos_err = 0.0 + m_cart.p.x();

        double f_angle = m_pid_angle.update(angle_err, dt);
        double f_pos = m_pid_pos.update(pos_err, dt);
        double force = f_angle + f_pos;

        m_control_force.set_force(Vector2d(force, 0));
        m_last_force = force;

        m_system.process(dt, SimSteps);
        m_time += dt;

        double ke = m_cart.energy() + m_pendulum.energy();
        double pe = m_pendulum.m * Gravity * m_pendulum.p.y() +
                    m_cart.m * Gravity * m_cart.p.y();

        push(m_angle_hist, m_pendulum.theta);
        push(m_cart_hist, m_cart.p.x());
        push(m_force_hist, force);
        push(m_energy_hist, ke + pe);
    }

    void render(Rendering::Renderer *r) {
        r->draw_grid(1.0, 50.0, Rendering::palette::grid_line(),
                     Rendering::palette::grid_axis());

        r->draw_line(-30, 0, 30, 0, 3.0f, Rendering::palette::grid_axis());

        // target marker
        r->draw_line(0, -0.3, 0, 0.3, 2.0f, Rendering::palette::accent3());

        double wr = 0.12;
        r->draw_circle(m_cart.p.x() - CartWidth * 0.35, -CartHeight * 0.5 - wr,
                       wr, Rendering::palette::text_dim());
        r->draw_circle(m_cart.p.x() + CartWidth * 0.35, -CartHeight * 0.5 - wr,
                       wr, Rendering::palette::text_dim());

        r->draw_bar(m_cart.p.x(), m_cart.p.y(), 0, CartWidth, CartHeight,
                    Rendering::palette::foreground(),
                    Rendering::palette::shadow());

        r->draw_bar(m_pendulum.p.x(), m_pendulum.p.y(), m_pendulum.theta,
                    PendulumLen, PendulumWidth, Rendering::palette::accent2(),
                    Rendering::palette::shadow());

        Vector2d piv;
        m_cart.local_to_world(Vector2d(0, 0), &piv);
        r->draw_circle(piv.x(), piv.y(), 0.09, Rendering::palette::accent1());

        Vector2d tip;
        m_pendulum.local_to_world(Vector2d(PendulumLen / 2.0, 0), &tip);
        r->draw_circle(tip.x(), tip.y(), 0.07, Rendering::palette::accent3());

        if (std::abs(m_last_force) > 1.0) {
            double s = 0.005;
            r->draw_arrow(m_cart.p.x(), m_cart.p.y() - CartHeight * 0.8,
                          m_cart.p.x() + m_last_force * s,
                          m_cart.p.y() - CartHeight * 0.8, 2.0f,
                          Rendering::palette::accent1());
        }

        render_hud(r);
        render_tuner(r);
        render_plots(r);
    }

    void handle_input(Rendering::Renderer *r) {
        // camera
        float wheel = r->get_mouse_wheel_move();
        if (wheel != 0.0f) {
            double z = r->camera_zoom() * ((wheel > 0) ? 1.1 : 1.0 / 1.1);
            r->set_camera(r->camera_x(), r->camera_y(),
                          std::clamp(z, 5.0, 500.0));
        }
        if (r->is_mouse_button_down(Rendering::mouse::Middle) ||
            r->is_mouse_button_down(Rendering::mouse::Right)) {
            float mdx, mdy;
            r->get_mouse_delta(&mdx, &mdy);
            double z = r->camera_zoom();
            r->set_camera(r->camera_x() - mdx / z, r->camera_y() + mdy / z, z);
        }
        if (r->is_key_pressed(Rendering::keys::H))
            r->set_camera(0.0, 1.5, 60.0);

        // simulation
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space))
            m_pendulum.v_theta += 2.0;

        // tuner navigation
        // Tab: switch angle/position PID
        // W/S: select Kp/Ki/Kd
        // A/D: decrease/increase value
        // Shift+A/D: fine adjust
        if (r->is_key_pressed(KEY_TAB)) {
            m_tuner_target = (m_tuner_target == TunerTarget::Angle)
                                 ? TunerTarget::Position
                                 : TunerTarget::Angle;
        }
        if (r->is_key_pressed(Rendering::keys::W)) {
            int p = (int)m_tuner_param;
            m_tuner_param = (TunerParam)std::max(0, p - 1);
        }
        if (r->is_key_pressed(Rendering::keys::S)) {
            int p = (int)m_tuner_param;
            m_tuner_param = (TunerParam)std::min(2, p + 1);
        }

        bool fine = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
        double step = fine ? 0.1 : 1.0;

        if (r->is_key_down(Rendering::keys::D)) {
            adjust_gain(step);
        }
        if (r->is_key_down(Rendering::keys::A)) {
            adjust_gain(-step);
        }
    }

  private:
    static double angle_error(double current, double target) {
        double err = target - current;
        while (err > M_PI)
            err -= 2.0 * M_PI;
        while (err < -M_PI)
            err += 2.0 * M_PI;
        return err;
    }

    static void push(std::deque<double> &h, double v) {
        h.push_back(v);
        while ((int)h.size() > MaxHistory)
            h.pop_front();
    }

    void adjust_gain(double delta) {
        double *target = nullptr;
        double scale = 1.0;

        if (m_tuner_target == TunerTarget::Angle) {
            switch (m_tuner_param) {
            case TunerParam::Kp:
                target = &m_kp_a;
                scale = 5.0;
                break;
            case TunerParam::Ki:
                target = &m_ki_a;
                scale = 0.5;
                break;
            case TunerParam::Kd:
                target = &m_kd_a;
                scale = 2.0;
                break;
            }
        } else {
            switch (m_tuner_param) {
            case TunerParam::Kp:
                target = &m_kp_p;
                scale = 0.5;
                break;
            case TunerParam::Ki:
                target = &m_ki_p;
                scale = 0.1;
                break;
            case TunerParam::Kd:
                target = &m_kd_p;
                scale = 0.5;
                break;
            }
        }

        if (target) {
            *target = std::max(0.0, *target + delta * scale);
            apply_gains();
        }
    }

    void apply_gains() {
        m_pid_angle.set_gains(m_kp_a, m_ki_a, m_kd_a);
        m_pid_angle.set_output_limits(-500.0, 500.0);
        m_pid_angle.set_integral_limits(-50.0, 50.0);
        m_pid_angle.reset();

        m_pid_pos.set_gains(m_kp_p, m_ki_p, m_kd_p);
        m_pid_pos.set_output_limits(-50.0, 50.0);
        m_pid_pos.set_integral_limits(-20.0, 20.0);
        m_pid_pos.reset();
    }

    void render_hud(Rendering::Renderer *r) {
        int x = 12, y = 12, sz = 18, dy = 22;
        char buf[128];

        r->draw_text("CART-PENDULUM PID", x, y, sz + 2,
                     Rendering::palette::accent2());
        y += dy + 6;

        double ke = m_cart.energy() + m_pendulum.energy();
        double pe = m_pendulum.m * Gravity * m_pendulum.p.y();

        std::snprintf(buf, sizeof(buf), "Angle:  %.2f deg",
                      m_pendulum.theta * 180.0 / M_PI);
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "Cart X: %.3f m", m_cart.p.x());
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "Force:  %.1f N", m_last_force);
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "Energy: %.2f J", ke + pe);
        r->draw_text(buf, x, y, sz, Rendering::palette::accent3());
        y += dy;
    }

    void render_tuner(Rendering::Renderer *r) {
        int x = 12, y = 220, sz = 16, dy = 20;
        char buf[128];

        auto dim = Rendering::palette::text_dim();
        auto sel = Rendering::palette::accent2();
        auto sel2 = Rendering::palette::accent3();
        auto hi = Rendering::palette::accent1();

        r->draw_text(
            "PID TUNER  [Tab] Switch  [W/S] Param  [A/D] Adjust  [Shift] Fine",
            x, y, sz - 3, dim);
        y += dy + 2;

        bool angle_sel = (m_tuner_target == TunerTarget::Angle);
        bool pos_sel = (m_tuner_target == TunerTarget::Position);

        // angle PID row
        auto angle_color = angle_sel ? sel : dim;
        r->draw_text(angle_sel ? "> Angle PID" : "  Angle PID", x, y, sz,
                     angle_color);
        y += dy;

        const char *param_names[] = {"Kp", "Ki", "Kd"};
        double angle_vals[] = {m_kp_a, m_ki_a, m_kd_a};

        for (int i = 0; i < 3; ++i) {
            bool this_sel = angle_sel && (int)m_tuner_param == i;
            auto c =
                this_sel ? hi : (angle_sel ? Rendering::palette::text() : dim);
            std::snprintf(buf, sizeof(buf), "    %s: %8.2f %s", param_names[i],
                          angle_vals[i], this_sel ? "<< >>" : "");
            r->draw_text(buf, x, y, sz, c);
            y += dy;
        }

        y += 4;

        // position PID row
        auto pos_color = pos_sel ? sel2 : dim;
        r->draw_text(pos_sel ? "> Position PID" : "  Position PID", x, y, sz,
                     pos_color);
        y += dy;

        double pos_vals[] = {m_kp_p, m_ki_p, m_kd_p};

        for (int i = 0; i < 3; ++i) {
            bool this_sel = pos_sel && (int)m_tuner_param == i;
            auto c =
                this_sel ? hi : (pos_sel ? Rendering::palette::text() : dim);
            std::snprintf(buf, sizeof(buf), "    %s: %8.2f %s", param_names[i],
                          pos_vals[i], this_sel ? "<< >>" : "");
            r->draw_text(buf, x, y, sz, c);
            y += dy;
        }

        y += 6;
        r->draw_text("[R] Reset   [SPACE] Perturb   [H] Home camera", x, y,
                     sz - 4, dim);
    }

    void render_plots(Rendering::Renderer *r) {
        int sw = r->screen_width();
        int pw = 280, ph = 80, margin = 12, gap = 5;
        int px = sw - pw - margin;

        draw_plot(
            r, px, margin, pw, ph, "Angle (deg)", m_angle_hist,
            [](double v) { return v * 180.0 / M_PI; },
            Rendering::palette::accent2());
        draw_plot(
            r, px, margin + ph + gap, pw, ph, "Cart X (m)", m_cart_hist,
            [](double v) { return v; }, Rendering::palette::accent3());
        draw_plot(
            r, px, margin + 2 * (ph + gap), pw, ph, "Force (N)", m_force_hist,
            [](double v) { return v; }, Rendering::palette::accent1());
        draw_plot(
            r, px, margin + 3 * (ph + gap), pw, ph, "Energy (J)", m_energy_hist,
            [](double v) { return v; }, Rendering::palette::foreground());
    }

    template <typename Fn>
    void draw_plot(Rendering::Renderer *r, int bx, int by, int bw, int bh,
                   const char *label, const std::deque<double> &data,
                   Fn transform, Rendering::Color lc) {
        r->draw_screen_rect(bx - 4, by - 4, bw + 8, bh + 8,
                            Rendering::palette::panel_bg());
        r->draw_text(label, bx + 2, by, 14, Rendering::palette::text_dim());
        int n = (int)data.size();
        if (n < 2)
            return;
        int py = by + 16, ph = bh - 18;
        if (ph <= 0)
            return;
        double vmin = 1e20, vmax = -1e20;
        for (double v : data) {
            double t = transform(v);
            vmin = std::min(vmin, t);
            vmax = std::max(vmax, t);
        }
        if (vmax - vmin < 1e-8) {
            vmin -= 0.5;
            vmax += 0.5;
        }
        double pad = (vmax - vmin) * 0.1;
        vmin -= pad;
        vmax += pad;
        if (vmin < 0 && vmax > 0) {
            int zy = py + (int)((vmax / (vmax - vmin)) * ph);
            r->draw_screen_line(bx, zy, bx + bw, zy, 1.0f,
                                Rendering::palette::grid_axis());
        }
        for (int i = 1; i < n; ++i) {
            double v0 = transform(data[i - 1]), v1 = transform(data[i]);
            r->draw_screen_line(
                bx + (i - 1) * bw / n,
                py + (int)((vmax - v0) / (vmax - vmin) * ph), bx + i * bw / n,
                py + (int)((vmax - v1) / (vmax - vmin) * ph), 1.5f, lc);
        }
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f", vmax);
        r->draw_text(buf, bx + bw - 45, py, 11, Rendering::palette::text_dim());
        std::snprintf(buf, sizeof(buf), "%.2f", vmin);
        r->draw_text(buf, bx + bw - 45, py + ph - 11, 11,
                     Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_cart, m_pendulum;
    Solver::LineConstraint m_rail;
    Solver::FixedRotationConstraint m_cart_rot;
    Solver::LinkConstraint m_pivot;
    Solver::GravityForceGenerator m_gravity;
    Solver::DirectForceGenerator m_control_force;

    Control::PIDController m_pid_angle;
    double m_kp_a = 150.0, m_ki_a = 2.0, m_kd_a = 40.0;

    Control::PIDController m_pid_pos;
    double m_kp_p = 3.0, m_ki_p = 0.2, m_kd_p = 5.0;

    double m_last_force = 0;

    TunerTarget m_tuner_target = TunerTarget::Angle;
    TunerParam m_tuner_param = TunerParam::Kp;

    std::deque<double> m_angle_hist, m_cart_hist, m_force_hist, m_energy_hist;
    double m_time = 0;
};

} // namespace manifold::Demo
