#pragma once

#include "manifold/renderer/theme.h"
#include "manifold/solver/forces/impulse.h"
#include <cstdio>
#include <manifold/control/ilqr.h>
#include <manifold/control/state_feedback.h>
#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/direct_force.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>
#include <raylib.h>

#include <algorithm>
#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
class CartDoublePendulumDemo : public DemoBase {
  public:
    // bottom-left of the slot's grid patch, which spans x[-4, 4] y[-2.9, 3.5]
    // in these coords -- see the cart slot in StoryDemo::build_frames
    static constexpr double ModeX = -2.3, ModeY = -2.42;
    static constexpr int ModeSize = 14;

    static constexpr double CartMass = 5.0;
    static constexpr double CartWidth = 1.2;
    static constexpr double CartHeight = 0.34;
    static constexpr double L1 = 1.30;
    static constexpr double L2 = 1.30;
    static constexpr double M1 = 0.9;
    static constexpr double M2 = 0.5;
    static constexpr double BarWidth = 0.085;
    static constexpr double BarWidthRenderCell = 0.075;
    static constexpr double Gravity = 9.81;
    static constexpr int SimSteps = 100;

    static constexpr double WheelR = 0.13;
    static constexpr double GroundY =
        -(CartHeight * 0.5 + 2.0 * WheelR + 0.035);

    static constexpr double UP = M_PI / 2.0;
    static constexpr double DOWN = -M_PI / 2.0;
    static constexpr double Sat = 300.0;

    using Vec6 = Eigen::Matrix<double, 6, 1>;

    enum class Stage { SwingUp1, Hold1, SwingUp2, Hold2 };

    void set_auto_kick(double first_at, double period, double impulse_ns,
                       double duration = 0.30) {
        m_kick_at = first_at;
        m_kick_period = period;
        m_kick_dur = std::max(1e-3, duration);
        m_kick_amp = impulse_ns / (m_kick_dur * 2.0 / M_PI);
    }

    // one disturbance per hold, so the reel shows the controller rejecting
    // something rather than just sitting still
    void set_stage_kicks(bool on) { m_stage_kicks = on; }

    const char *name() const override { return "Cart Double-Pendulum"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 1.5; }
    double default_cam_zoom() const override { return 60.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_cart.reset();
        m_cart.m = CartMass;
        m_cart.I =
            CartMass * (CartWidth * CartWidth + CartHeight * CartHeight) / 12.0;
        m_system.add_body(&m_cart);

        m_bar1.reset();
        m_bar1.m = M1;
        m_bar1.I = M1 * L1 * L1 / 12.0;
        m_system.add_body(&m_bar1);

        m_bar2.reset();
        m_bar2.m = M2;
        m_bar2.I = M2 * L2 * L2 / 12.0;
        m_system.add_body(&m_bar2);

        place(0.0, 0.0, -UP + 0.02, 0.0, -UP, 0.0); // hanging at rest

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

        m_pivot.set_bodies(&m_cart, &m_bar1);
        m_pivot.set_local_pos1(Vector2d(0, 0));
        m_pivot.set_local_pos2(Vector2d(-L1 / 2.0, 0));
        m_pivot.set_ks(100.0);
        m_pivot.set_kd(10.0);
        m_system.add_constraint(&m_pivot);

        m_elbow.set_bodies(&m_bar1, &m_bar2);
        m_elbow.set_local_pos1(Vector2d(L1 / 2.0, 0));
        m_elbow.set_local_pos2(Vector2d(-L2 / 2.0, 0));
        m_elbow.set_ks(100.0);
        m_elbow.set_kd(10.0);
        m_system.add_constraint(&m_elbow);

        m_gravity.set_gravity(Gravity);
        m_system.add_force_generator(&m_gravity);

        m_control_force.set_body(&m_cart);
        m_system.add_force_generator(&m_control_force);

        m_kick.set_body(&m_bar2);
        m_system.add_force_generator(&m_kick);

        m_lqr.set_output_limits(-Sat, Sat);

        m_plot_a1.configure("Link 1 (deg)", Rendering::palette::accent2(), 600,
                            [](double v) { return v * 180.0 / M_PI; });
        m_plot_a2.configure("Link 2 (deg)", Rendering::palette::accent4(), 600,
                            [](double v) { return v * 180.0 / M_PI; });
        m_plot_cart.configure("Cart X (m)", Rendering::palette::accent3());
        m_plot_force.configure("Force (N)", Rendering::palette::accent1());
        m_plot_a1.clear();
        m_plot_a2.clear();
        m_plot_cart.clear();
        m_plot_force.clear();

        m_t = 0.0;
        m_stage_t = 0.0;
        m_last_force = 0.0;
        enter(Stage::SwingUp1);
        m_force_shown = m_force_vel = m_force_peak = 0.0;
        m_kick_shown = m_kick_vel = m_kick_force = 0.0;
        m_kick_ref = 1.0;
        m_kick_until = -1.0;

        clear_overlays();
        Rendering::register_constraint_overlays(*this, m_system);
    }

    void process(double dt) override {
        m_t += dt;
        m_stage_t += dt;
        m_since_replan += dt;

        if (m_kick_at > 0.0 && m_t >= m_kick_at) {
            fire_kick_amp(m_kick_amp, m_kick_dur);
            m_kick_at = m_kick_period > 0.0 ? m_t + m_kick_period : 0.0;
        }
        if (m_stage_kick_ns != 0.0 && !m_stage_kicked &&
            m_stage_t >= m_stage_kick_at) {
            m_stage_kicked = true;
            fire_kick(m_stage_kick_ns, KickDur);
        }

        m_kick_force = 0.0;
        if (m_t < m_kick_until) {
            const double u = 1.0 - (m_kick_until - m_t) / m_kick_dur;
            m_kick_force =
                m_kick_amp * std::sin(M_PI * std::clamp(u, 0.0, 1.0));
            m_kick.arm_force(Vector2d(m_kick_force, 0.0));
        }

        const double force = control();
        m_control_force.set_force(Vector2d(force, 0));
        m_last_force = force;

        ramp(m_last_force, dt, &m_force_shown, &m_force_vel);
        ramp(m_kick_force, dt, &m_kick_shown, &m_kick_vel);
        m_force_peak = std::max(std::abs(m_force_shown),
                                m_force_peak * std::exp(-0.25 * dt));

        m_system.process(dt, SimSteps);
        m_kick.disarm();

        m_plot_a1.push(wrap(m_bar1.theta - UP));
        m_plot_a2.push(wrap(m_bar2.theta - UP));
        m_plot_cart.push(m_cart.p.x());
        m_plot_force.push(force);
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        render_cell(r);
        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_a1, &m_plot_a2, &m_plot_cart,
                                           &m_plot_force};
        render_plots(r, plots);
    }

    void render_cell(Rendering::Renderer *r) override {
        const auto fg = Rendering::palette::foreground();
        const auto dim = Rendering::palette::text_dim();
        const auto a1 = Rendering::palette::accent1();
        const auto a2 = Rendering::palette::accent2();
        const auto a3 = Rendering::palette::accent3();
        const auto blue = Rendering::palette::accent4();

        const double wy = -CartHeight * 0.5 - WheelR - 0.035;
        const double a = -m_cart.p.x() / WheelR;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            const double wx = m_cart.p.x() + sgn * CartWidth * 0.34;
            Rendering::draw_body_disk(r, wx, wy, WheelR, -a,
                                      {.fill = dim, .show_shadow = true});
            r->draw_circle(wx + 0.55 * WheelR * std::cos(a),
                           wy + 0.55 * WheelR * std::sin(a), 0.028, fg);
        }

        Rendering::Layer l = r->current_layer();
        r->set_layer(Rendering::Layer::Content);
        if (std::abs(m_cart.p.x()) > 0.02)
            Rendering::draw_displacement(r, -0.03, GroundY, m_cart.p.x(),
                                         GroundY, "x = %.2f m", m_cart.p.x(),
                                         2.0f, a1, 0.0, {.offset = -0.62});

        const double ref = std::max(m_force_peak, 1.0);
        const double u = std::clamp(std::abs(m_force_shown) / ref, 0.0, 1.0);
        if (u > 0.02) {
            const double len = CartWidth * (0.09 + 1.15 * std::pow(u, 0.7));
            const double dir = m_force_shown > 0 ? 1.0 : -1.0;
            const double x0 = m_cart.p.x() + dir * (CartWidth * 0.5 + 0.14);
            const double ay = m_cart.p.y();
            draw_vector(r, x0, ay, dir * len, 0.0, 3.0f, a2);

            Rendering::LayerScope txt(r, Rendering::Layer::Text);
            char buf[24];
            std::snprintf(buf, sizeof buf, "%.0f N", m_force_shown);
            int sx, sy;
            r->world_to_screen(x0 + dir * len, ay, &sx, &sy);
            const int tw = r->measure_text(buf, 11);
            r->draw_text(buf, dir > 0 ? sx + 7 : sx - tw - 7, sy - 6, 11, a2);
        }

        // the kick rides its own fixed reference, not a decaying peak: an
        // isolated pulse would pin a peak-normalised arrow at full length from
        // the first frame, which is the pop we are trying to avoid
        const double ku = std::clamp(
            std::abs(m_kick_shown) / std::max(m_kick_ref, 1e-3), 0.0, 1.0);
        if (ku > 0.02) {
            const double len = L2 * (0.06 + 0.55 * std::pow(ku, 0.7));
            const double dir = m_kick_shown > 0 ? 1.0 : -1.0;
            const double x0 = m_bar2.p.x() + dir * 0.14;
            const double ay = m_bar2.p.y();
            draw_vector(r, x0, ay, dir * len, 0.0, 3.0f, blue);

            Rendering::LayerScope txt(r, Rendering::Layer::Text);
            char buf[24];
            std::snprintf(buf, sizeof buf, "%.0f N", m_kick_shown);
            int sx, sy;
            r->world_to_screen(x0 + dir * len, ay, &sx, &sy);
            const int tw = r->measure_text(buf, 11);
            r->draw_text(buf, dir > 0 ? sx + 7 : sx - tw - 7, sy - 6, 11, blue);
        }
        r->set_layer(l);

        Rendering::draw_body_block_circular(r, m_cart.p, CartWidth, CartHeight,
                                            0.0, {.show_shadow = true});

        Vector2d piv, elbow, tip;
        m_cart.local_to_world(Vector2d(0, 0), &piv);
        m_bar1.local_to_world(Vector2d(L1 / 2.0, 0), &elbow);
        m_bar2.local_to_world(Vector2d(L2 / 2.0, 0), &tip);

        Rendering::draw_body_bar(r, m_bar1.p, L1, BarWidthRenderCell,
                                 m_bar1.theta,
                                 {.show_center = false, .show_shadow = true});

        // needs to draw underneath
        Rendering::draw_body_disk(
            r, piv, 0.07, 0,
            {.fill = a1, .show_center = false, .show_shadow = true});

        Rendering::draw_body_bar(r, m_bar2.p, L2, BarWidthRenderCell,
                                 m_bar2.theta,
                                 {.show_center = false, .show_shadow = true});

        // tip
        Rendering::draw_body_disk(r, tip, 0.13, m_bar2.theta,
                                  {.fill = a3, .show_shadow = true});
        Rendering::draw_body_disk(
            r, elbow, 0.07, 0,
            {.fill = blue, .show_center = false, .show_shadow = true});

        draw_mode(r);
    }

    // Names what the controller is doing, in the bottom-left of the grid patch.
    // stage_name() is the HUD's version and too long to sit under the art; this
    // one is the beat rather than the method, and a live disturbance overrides
    // the stage because that is what you are actually watching in that moment.
    const char *cell_mode() const {
        if (m_t < m_kick_until)
            return "PERTURBATION";
        switch (m_stage) {
        case Stage::SwingUp1:
            return "KICK-UP 1";
        case Stage::Hold1:
            return "BALANCE";
        case Stage::SwingUp2:
            return "KICK-UP 2";
        case Stage::Hold2:
            return "BALANCE";
        }
        return "";
    }

    void draw_mode(Rendering::Renderer *r) const {
        Rendering::LayerScope txt(r, Rendering::Layer::Text);
        int sx, sy;
        r->world_to_screen(ModeX, ModeY, &sx, &sy);
        r->draw_text(cell_mode(), sx, sy, ModeSize,
                     Rendering::palette::text_dim());
    }

    static void draw_vector(Rendering::Renderer *r, double x0, double y0,
                            double dx, double dy, float thickness,
                            Rendering::Color c) {
        const double len = std::hypot(dx, dy);
        if (len < 1e-6)
            return;
        const double ux = dx / len, uy = dy / len;
        const double head = std::clamp(0.22 * len, 0.06, 0.28);
        const double hw = head * 0.46;
        const double bx = x0 + dx - ux * head, by = y0 + dy - uy * head;
        r->draw_line(x0, y0, bx, by, thickness, c);
        r->draw_triangle(x0 + dx, y0 + dy, bx - uy * hw, by + ux * hw,
                         bx + uy * hw, by - ux * hw, c);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space)) {
            const bool rev = r->is_key_down(Rendering::keys::LeftShift) ||
                             r->is_key_down(Rendering::keys::RightShift);
            fire_kick_amp((rev ? -1.0 : 1.0) * 26.0, m_kick_dur);
        }
    }

  private:
    // second-order lag, so an arrow never steps
    static void ramp(double target, double dt, double *shown, double *vel) {
        constexpr double wn = 14.0;
        *vel += (wn * wn * (target - *shown) - 2.0 * wn * *vel) * dt;
        *shown += *vel * dt;
    }

    void fire_kick_amp(double amp, double dur) {
        m_kick_dur = std::max(1e-3, dur);
        m_kick_amp = amp;
        m_kick_ref = std::max(std::abs(amp) * KickShownGain, 1e-3);
        m_kick_until = m_t + m_kick_dur;
    }

    void fire_kick(double impulse_ns, double dur) {
        fire_kick_amp(impulse_ns / (std::max(1e-3, dur) * 2.0 / M_PI), dur);
    }

    static double wrap(double a) {
        while (a > M_PI)
            a -= 2.0 * M_PI;
        while (a < -M_PI)
            a += 2.0 * M_PI;
        return a;
    }

    void place(double x, double xd, double t1, double t1d, double t2,
               double t2d) {
        m_cart.p = Vector2d(x, 0.0);
        m_cart.v = Vector2d(xd, 0.0);
        m_cart.theta = 0.0;
        m_cart.v_theta = 0.0;

        const Vector2d r1(std::cos(t1), std::sin(t1));
        const Vector2d n1(-std::sin(t1), std::cos(t1));
        m_bar1.theta = t1;
        m_bar1.v_theta = t1d;
        m_bar1.p = m_cart.p + r1 * (L1 / 2.0);
        m_bar1.v = m_cart.v + n1 * (t1d * L1 / 2.0);

        const Vector2d root2 = m_cart.p + r1 * L1;
        const Vector2d vroot2 = m_cart.v + n1 * (t1d * L1);
        const Vector2d r2(std::cos(t2), std::sin(t2));
        const Vector2d n2(-std::sin(t2), std::cos(t2));
        m_bar2.theta = t2;
        m_bar2.v_theta = t2d;
        m_bar2.p = root2 + r2 * (L2 / 2.0);
        m_bar2.v = vroot2 + n2 * (t2d * L2 / 2.0);
    }

    Vec6 error_state() const { return goal_error(state(), UP, UP); }

    Vec6 state() const {
        Vec6 s;
        s << m_cart.p.x(), m_cart.v.x(), m_bar1.theta, m_bar1.v_theta,
            m_bar2.theta, m_bar2.v_theta;
        return s;
    }

    static Vec6 goal_error(const Vec6 &s, double t1g, double t2g) {
        Vec6 e;
        e << s(0), s(1), wrap(s(2) - t1g), s(3), wrap(s(4) - t2g), s(5);
        return e;
    }

    static Vec6 model(const Vec6 &s, double u) {
        constexpr double a1 = L1 / 2.0, a2 = L2 / 2.0;
        constexpr double In1 = M1 * L1 * L1 / 12.0, In2 = M2 * L2 * L2 / 12.0;
        constexpr double k12 = M1 * a1 + M2 * L1, k13 = M2 * a2,
                         k23 = M2 * L1 * a2;
        constexpr double m11 = CartMass + M1 + M2,
                         m22 = M1 * a1 * a1 + In1 + M2 * L1 * L1,
                         m33 = M2 * a2 * a2 + In2;
        const double t1 = s(2), w1 = s(3), t2 = s(4), w2 = s(5);
        const double s1 = std::sin(t1), s2 = std::sin(t2);
        const double sd = std::sin(t1 - t2), cd = std::cos(t1 - t2);

        Eigen::Matrix3d M;
        M << m11, -k12 * s1, -k13 * s2, -k12 * s1, m22, k23 * cd, -k13 * s2,
            k23 * cd, m33;
        Eigen::Vector3d rhs;
        rhs << u + k12 * std::cos(t1) * w1 * w1 + k13 * std::cos(t2) * w2 * w2,
            -k23 * sd * w2 * w2 - k12 * Gravity * std::cos(t1),
            k23 * sd * w1 * w1 - k13 * Gravity * std::cos(t2);
        const Eigen::Vector3d a = M.ldlt().solve(rhs);

        Vec6 d;
        d << s(1), a(0), w1, a(1), w2, a(2);
        return d;
    }

    void configure_mpc(double t1g, double t2g, double seed_amp, double w1,
                       double w2) {
        m_mpc.dynamics = [](const Vec6 &x, double u) { return model(x, u); };
        m_mpc.err = [t1g, t2g](const Vec6 &x) {
            Vec6 e;
            e << x(0), x(1), 2.0 * std::sin(0.5 * wrap(x(2) - t1g)), x(3),
                2.0 * std::sin(0.5 * wrap(x(4) - t2g)), x(5);
            return e;
        };
        m_mpc.q << 14.0, 1.6, 26.0 * w1, 0.9 * w1, 26.0 * w2, 0.9 * w2;
        m_mpc.qf << 90.0, 26.0, 900.0 * w1, 40.0 * w1, 900.0 * w2, 40.0 * w2;
        m_mpc.r = 0.012;
        m_mpc.u_min = -Sat;
        m_mpc.u_max = Sat;
        m_mpc.dt = 0.05;
        m_mpc.iters = WarmIters;
        if (seed_amp > 0.0) {
            const double w = std::sqrt(3.0 * Gravity / (2.0 * L2));
            m_mpc.seed = [w, seed_amp, this](int i) {
                return seed_amp * std::sin(w * i * m_mpc.dt);
            };
        } else {
            m_mpc.seed = nullptr;
        }
        m_mpc.reset();
    }

    void enter(Stage s) {
        m_stage = s;
        m_stage_t = 0.0;
        m_since_replan = 0.0;
        m_stage_kicked = false;
        m_stage_kick_at = 0.0;
        m_stage_kick_ns = 0.0;
        switch (s) {
        case Stage::SwingUp1:
            start_swing(DOWN, UP, ManeuverT1, Kick1Seed, 1.0, Kick1Keep2);
            break;
        case Stage::SwingUp2:
            start_swing(UP, UP, ManeuverT2, Kick2Seed, 1.0, Kick2Keep2);
            break;
        case Stage::Hold1:
            m_lqr.set_gains(K1);
            if (m_stage_kicks) {
                m_stage_kick_at = Hold1KickAt;
                m_stage_kick_ns = Hold1KickNs;
            }
            break;
        case Stage::Hold2:
            m_lqr.set_gains(K2);
            if (m_stage_kicks) {
                m_stage_kick_at = Hold2KickAt;
                m_stage_kick_ns = Hold2KickNs;
            }
            break;
        }
    }

    void start_swing(double t1g, double t2g, double T, double seed_amp,
                     double w1, double w2) {
        m_maneuver = T;
        configure_mpc(t1g, t2g, seed_amp, w1, w2);
        m_mpc.horizon = (int)std::lround(T / m_mpc.dt);
        m_mpc.reset();
        m_mpc.solve(state(), ColdIters);
    }

    double swing_action() {
        const int n = (int)std::lround((m_maneuver - m_stage_t) / m_mpc.dt);
        while (m_since_replan >= m_mpc.dt) {
            m_since_replan -= m_mpc.dt;
            m_mpc.shift();
            m_mpc.horizon = std::max(20, n);
            m_mpc.solve(state(), WarmIters);
        }
        return m_mpc.action(state());
    }

    double control() {
        switch (m_stage) {
        case Stage::SwingUp1: {
            const Vec6 e = goal_error(state(), DOWN, UP);
            if (std::abs(e(4)) < 0.12 && std::abs(e(5)) < 0.70 &&
                std::abs(e(2)) < 0.45 && std::abs(e(3)) < 1.60) {
                enter(Stage::Hold1);
                return m_lqr.update(e);
            }
            return swing_action();
        }
        case Stage::Hold1: {
            const Vec6 e = goal_error(state(), DOWN, UP);
            if (std::abs(e(4)) > 0.60)
                enter(Stage::SwingUp1);
            else if (m_stage_t > HoldT)
                enter(Stage::SwingUp2);
            return m_lqr.update(e);
        }
        case Stage::SwingUp2: {
            const Vec6 e = error_state();
            if (std::abs(e(2)) < 0.12 && std::abs(e(4)) < 0.12 &&
                std::abs(e(3)) < 1.50 && std::abs(e(5)) < 0.65) {
                enter(Stage::Hold2);
                return m_lqr.update(e);
            }
            return swing_action();
        }
        case Stage::Hold2: {
            const Vec6 e = error_state();
            if (std::abs(e(2)) > 0.60 || std::abs(e(4)) > 0.70)
                enter(Stage::SwingUp1);
            return m_lqr.update(e);
        }
        }
        return 0.0;
    }

    const char *stage_name() const {
        switch (m_stage) {
        case Stage::SwingUp1:
            return "DDP kick 1 (link 2)";
        case Stage::Hold1:
            return "LQR hold (link 2 up)";
        case Stage::SwingUp2:
            return "DDP kick 2 (link 1)";
        case Stage::Hold2:
            return "LQR hold (both up)";
        }
        return "";
    }

    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("CART DOUBLE-PENDULUM", Rendering::palette::accent2());
        hud.small_text("iLQR MPC swing-up . LQR hold",
                       Rendering::palette::text());
        hud.line(Rendering::palette::accent3(), "stage:  %s", stage_name());
        hud.line(Rendering::palette::text(), "link 1: %+6.2f deg",
                 wrap(m_bar1.theta - UP) * 180.0 / M_PI);
        hud.line(Rendering::palette::text(), "link 2: %+6.2f deg",
                 wrap(m_bar2.theta - UP) * 180.0 / M_PI);
        hud.line(Rendering::palette::text(), "cart x: %+6.3f m", m_cart.p.x());
        hud.line(Rendering::palette::accent1(), "force:  %+6.1f N",
                 m_last_force);
        hud.separator();
        hud.small_text("[R] reset   [SPACE] disturb",
                       Rendering::palette::text_dim());
    }

    static const Vec6 K1, K2;
    double m_maneuver = 4.0;

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_cart, m_bar1, m_bar2;
    Solver::LineConstraint m_rail;
    Solver::FixedRotationConstraint m_cart_rot;
    Solver::LinkConstraint m_pivot, m_elbow;
    Solver::UniformGravityForceGenerator m_gravity;
    Solver::DirectForceGenerator m_control_force;
    Solver::ImpulseForceGenerator m_kick;

    Control::StateFeedback<6> m_lqr;
    Control::ILQR<6> m_mpc;
    static constexpr double ManeuverT1 = 3.0, ManeuverT2 = 3.0;
    static constexpr double HoldT = 2.0;
    // sized against the hold abort thresholds: both land a ~25 deg excursion
    // the LQR rejects. Hold1 is the tighter of the two — link 1 hangs, so K1
    // has far less authority over link 2 than K2 has with both up
    static constexpr double KickDur = 0.30;
    // the display lag peaks at ~0.69 of the applied amplitude for a KickDur
    // half-sine, so the reference is scaled to match or the arrow would top out
    // at two thirds of its length
    static constexpr double KickShownGain = 0.69;
    static constexpr double Hold1KickAt = 0.7, Hold1KickNs = 0.4;
    static constexpr double Hold2KickAt = 1.2, Hold2KickNs = 0.7;
    static constexpr double Kick1Seed = 0.0, Kick1Keep2 = 1.0;
    static constexpr double Kick2Seed = 30.0, Kick2Keep2 = 6.0;
    static constexpr int ColdIters = 80, WarmIters = 14;

    Stage m_stage = Stage::SwingUp1;
    double m_stage_t = 0.0;
    double m_since_replan = 0.0;

    double m_t = 0.0, m_last_force = 0.0;
    double m_kick_at = 0.0, m_kick_period = 0.0;
    double m_kick_dur = 0.30, m_kick_amp = 0.0, m_kick_until = -1.0;
    double m_kick_force = 0.0, m_kick_ref = 1.0;
    double m_kick_shown = 0.0, m_kick_vel = 0.0;
    double m_force_shown = 0.0, m_force_vel = 0.0, m_force_peak = 0.0;

    bool m_stage_kicks = false, m_stage_kicked = false;
    double m_stage_kick_at = 0.0, m_stage_kick_ns = 0.0;

    PlotWidget m_plot_a1, m_plot_a2, m_plot_cart, m_plot_force;
};

inline const CartDoublePendulumDemo::Vec6 CartDoublePendulumDemo::K1 =
    (CartDoublePendulumDemo::Vec6() << -10.4447, -22.7455, -1008.1902,
     -216.6775, 1288.3190, 312.9092)
        .finished();
inline const CartDoublePendulumDemo::Vec6 CartDoublePendulumDemo::K2 =
    (CartDoublePendulumDemo::Vec6() << 10.4447, 26.5457, 615.5468, -3.4580,
     -858.0753, -198.5813)
        .finished();

} // namespace manifold::Demo
