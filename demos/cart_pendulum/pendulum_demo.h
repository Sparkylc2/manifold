#pragma once

#include <manifold/renderer/renderer.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>

namespace manifold::Demo {
using Vector2d = Eigen::Vector2d;
class PendulumDemo {
  public:
    static constexpr double AnchorMass = 1e6;
    static constexpr double AnchorInertia = 1e6;
    static constexpr double BobMass = 1.0;
    static constexpr double BarLen = 3.0;
    static constexpr double BarWidth = 0.1;
    static constexpr double Gravity = 9.81;
    static constexpr int SimSteps = 100;
    static constexpr int MaxHistory = 600;

    void initialize() {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        // anchor body: massive, pinned at origin
        m_anchor.reset();
        m_anchor.m = AnchorMass;
        m_anchor.I = AnchorInertia;
        m_anchor.p = Vector2d(0, 0);
        m_system.add_body(&m_anchor);

        m_pin.set_body(&m_anchor);
        m_pin.set_world_position(Vector2d(0, 0));
        m_pin.set_local_position(Vector2d(0, 0));
        m_pin.set_ks(100.0);
        m_pin.set_kd(10.0);
        m_system.add_constraint(&m_pin);

        // pendulum bar: link at local(+L/2, 0) pinned to anchor
        // hanging equilibrium: theta = pi/2 (bar points up in local x,
        // but rotated so local x points down in world)
        // start 45 deg from hanging
        double angle = M_PI / 2.0 + M_PI / 4.0;

        m_bar.reset();
        m_bar.m = BobMass;
        m_bar.I = BobMass * BarLen * BarLen / 12.0;
        m_bar.theta = angle;
        m_bar.p = Vector2d(-std::cos(angle) * BarLen / 2.0,
                           -std::sin(angle) * BarLen / 2.0);
        m_system.add_body(&m_bar);

        m_link.set_bodies(&m_anchor, &m_bar);
        m_link.set_local_pos1(Vector2d(0, 0));
        m_link.set_local_pos2(Vector2d(BarLen / 2.0, 0));
        m_link.set_ks(100.0);
        m_link.set_kd(10.0);
        m_system.add_constraint(&m_link);

        m_gravity.set_gravity(Gravity);
        m_system.add_force_generator(&m_gravity);

        m_energy_hist.clear();
        m_angle_hist.clear();
        m_drift_hist.clear();
        m_time = 0;
    }

    void process(double dt) {
        m_system.process(dt, SimSteps);
        m_time += dt;

        double ke = m_anchor.energy() + m_bar.energy();
        double pe = m_bar.m * Gravity * m_bar.p.y();
        double total = ke + pe;

        Vector2d bar_top;
        m_bar.local_to_world(Vector2d(BarLen / 2.0, 0), &bar_top);
        double drift = (bar_top - m_anchor.p).norm();

        push(m_energy_hist, total);
        push(m_angle_hist, m_bar.theta);
        push(m_drift_hist, drift);
    }

    void render(Rendering::Renderer *r) {
        r->draw_grid(1.0, 50.0, Rendering::palette::grid_line(),
                     Rendering::palette::grid_axis());

        // anchor
        r->draw_circle(m_anchor.p.x(), m_anchor.p.y(), 0.15,
                       Rendering::palette::accent1());

        // bar
        r->draw_bar(m_bar.p.x(), m_bar.p.y(), m_bar.theta, BarLen, BarWidth,
                    Rendering::palette::accent2(),
                    Rendering::palette::shadow());

        // linked end (should sit on anchor)
        Vector2d bar_top;
        m_bar.local_to_world(Vector2d(BarLen / 2.0, 0), &bar_top);
        r->draw_circle(bar_top.x(), bar_top.y(), 0.06,
                       Rendering::palette::accent3());

        // tip
        Vector2d bar_bot;
        m_bar.local_to_world(Vector2d(-BarLen / 2.0, 0), &bar_bot);
        r->draw_circle(bar_bot.x(), bar_bot.y(), 0.08,
                       Rendering::palette::foreground());

        render_hud(r);
        render_plots(r);
    }

    void handle_input(Rendering::Renderer *r) {
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
            r->set_camera(0.0, -1.0, 60.0);
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space))
            m_bar.v_theta += 3.0;
    }

  private:
    static void push(std::deque<double> &h, double v) {
        h.push_back(v);
        while ((int)h.size() > MaxHistory)
            h.pop_front();
    }

    void render_hud(Rendering::Renderer *r) {
        int x = 12, y = 12, sz = 18, dy = 22;
        char buf[128];

        r->draw_text("PENDULUM", x, y, sz + 2, Rendering::palette::accent2());
        y += dy + 6;

        double ke = m_anchor.energy() + m_bar.energy();
        double pe = m_bar.m * Gravity * m_bar.p.y();

        Vector2d bar_top;
        m_bar.local_to_world(Vector2d(BarLen / 2.0, 0), &bar_top);
        double drift = (bar_top - m_anchor.p).norm();

        std::snprintf(buf, sizeof(buf), "Angle:  %.2f deg",
                      m_bar.theta * 180.0 / M_PI);
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "KE:     %.4f J", ke);
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "PE:     %.4f J", pe);
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "Total:  %.4f J", ke + pe);
        r->draw_text(buf, x, y, sz, Rendering::palette::accent3());
        y += dy;

        std::snprintf(buf, sizeof(buf), "Drift:  %.6f m", drift);
        auto dc = drift > 0.01 ? Rendering::palette::accent1()
                               : Rendering::palette::text();
        r->draw_text(buf, x, y, sz, dc);
        y += dy;

        y += 8;
        r->draw_text("[R] Reset  [SPACE] Kick  [H] Home", x, y, sz - 4,
                     Rendering::palette::text_dim());
    }

    void render_plots(Rendering::Renderer *r) {
        int sw = r->screen_width();
        int pw = 280, ph = 90, margin = 12, gap = 6;
        int px = sw - pw - margin;

        draw_plot(
            r, px, margin, pw, ph, "Angle (deg)", m_angle_hist,
            [](double v) { return v * 180.0 / M_PI; },
            Rendering::palette::accent2());
        draw_plot(
            r, px, margin + ph + gap, pw, ph, "Total Energy (J)", m_energy_hist,
            [](double v) { return v; }, Rendering::palette::accent3());
        draw_plot(
            r, px, margin + 2 * (ph + gap), pw, ph, "Constraint Drift (m)",
            m_drift_hist, [](double v) { return v; },
            Rendering::palette::accent1());
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
        std::snprintf(buf, sizeof(buf), "%.3f", vmax);
        r->draw_text(buf, bx + bw - 50, py, 11, Rendering::palette::text_dim());
        std::snprintf(buf, sizeof(buf), "%.3f", vmin);
        r->draw_text(buf, bx + bw - 50, py + ph - 11, 11,
                     Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_anchor, m_bar;
    Solver::FixedPositionConstraint m_pin;
    Solver::LinkConstraint m_link;
    Solver::GravityForceGenerator m_gravity;

    std::deque<double> m_energy_hist, m_angle_hist, m_drift_hist;
    double m_time = 0;
};

} // namespace manifold::Demo
