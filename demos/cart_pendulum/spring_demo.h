#pragma once

#include <manifold/renderer/renderer.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>

namespace manifold::Demo {
using Vector2d = Eigen::Vector2d;
class SpringDemo {
  public:
    static constexpr double Mass = 2.0;
    static constexpr double BoxSize = 0.6;
    static constexpr double SpringK = 20.0;
    static constexpr double SpringDamp = 0.0;
    static constexpr double StartX = 2.0;
    static constexpr int SimSteps = 100;
    static constexpr int MaxHistory = 600;

    void initialize() {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        // sliding mass
        m_body.reset();
        m_body.m = Mass;
        m_body.I = Mass * BoxSize * BoxSize / 6.0;
        m_body.p = Vector2d(StartX, 0);
        m_system.add_body(&m_body);

        // horizontal rail
        m_rail.set_body(&m_body);
        m_rail.set_line(Vector2d(0, 0), Vector2d(1, 0));
        m_rail.set_local_pos(Vector2d(0, 0));
        m_rail.set_ks(100.0);
        m_rail.set_kd(10.0);
        m_system.add_constraint(&m_rail);

        // no rotation
        m_no_rot.set_body(&m_body);
        m_no_rot.set_angle(0);
        m_no_rot.set_ks(100.0);
        m_no_rot.set_kd(10.0);
        m_system.add_constraint(&m_no_rot);

        // fixed anchor body (not added to system, index stays -1)
        m_anchor.reset();
        m_anchor.m = 1.0;
        m_anchor.I = 1.0;
        m_anchor.p = Vector2d(0, 0);
        // NOT added to system — acts as fixed point

        // spring between anchor and mass
        m_spring.set_bodies(&m_anchor, &m_body);
        m_spring.set_local_pos1(Vector2d(0, 0));
        m_spring.set_local_pos2(Vector2d(0, 0));
        m_spring.set_rest_length(0.0);
        m_spring.set_ks(SpringK);
        m_spring.set_kd(SpringDamp);
        m_system.add_force_generator(&m_spring);

        // no gravity

        m_pos_hist.clear();
        m_vel_hist.clear();
        m_energy_hist.clear();
        m_time = 0;
        m_initial_energy = 0.5 * SpringK * StartX * StartX;
    }

    void process(double dt) {
        m_system.process(dt, SimSteps);
        m_time += dt;

        double pe = m_spring.energy();
        double ke = m_body.energy();

        push(m_pos_hist, m_body.p.x());
        push(m_vel_hist, m_body.v.x());
        push(m_energy_hist, pe + ke);
    }

    void render(Rendering::Renderer *r) {
        r->draw_grid(1.0, 50.0, Rendering::palette::grid_line(),
                     Rendering::palette::grid_axis());

        // rail
        r->draw_line(-10, 0, 10, 0, 2.0f, Rendering::palette::grid_axis());

        // equilibrium
        r->draw_circle(0, 0, 0.05, Rendering::palette::text_dim());

        // spring visualization
        draw_spring_visual(r, 0, 0, m_body.p.x(), 0, 10, 0.25);

        // mass
        r->draw_bar(m_body.p.x(), m_body.p.y(), 0, BoxSize, BoxSize,
                    Rendering::palette::accent2(),
                    Rendering::palette::shadow());

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
            r->set_camera(0.0, 0.0, 80.0);
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space))
            m_body.v.x() += 3.0;
    }

  private:
    static void push(std::deque<double> &h, double v) {
        h.push_back(v);
        while ((int)h.size() > MaxHistory)
            h.pop_front();
    }

    void draw_spring_visual(Rendering::Renderer *r, double x0, double y0,
                            double x1, double y1, int coils, double amp) {
        double dx = x1 - x0, dy = y1 - y0;
        double len = std::sqrt(dx * dx + dy * dy);
        if (len < 0.01)
            return;
        double nx = dx / len, ny = dy / len;
        double px = -ny, py = nx;

        int segs = coils * 4;
        double prev_x = x0, prev_y = y0;
        for (int i = 1; i <= segs; ++i) {
            double t = (double)i / segs;
            double across = 0;
            int phase = i % 4;
            if (phase == 1)
                across = amp;
            else if (phase == 3)
                across = -amp;

            double cx = x0 + dx * t + px * across;
            double cy = y0 + dy * t + py * across;
            r->draw_line(prev_x, prev_y, cx, cy, 2.0f,
                         Rendering::palette::foreground());
            prev_x = cx;
            prev_y = cy;
        }
    }

    void render_hud(Rendering::Renderer *r) {
        int x = 12, y = 12, sz = 18, dy = 22;
        char buf[128];

        r->draw_text("SPRING OSCILLATOR", x, y, sz + 2,
                     Rendering::palette::accent2());
        y += dy + 6;

        double pe = m_spring.energy();
        double ke = m_body.energy();
        double total = pe + ke;
        double drift_pct =
            (m_initial_energy > 0)
                ? std::abs(total - m_initial_energy) / m_initial_energy * 100.0
                : 0.0;

        std::snprintf(buf, sizeof(buf), "Pos:    %.4f m", m_body.p.x());
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "Vel:    %.4f m/s", m_body.v.x());
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "KE:     %.4f J", ke);
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "PE:     %.4f J", pe);
        r->draw_text(buf, x, y, sz, Rendering::palette::text());
        y += dy;

        std::snprintf(buf, sizeof(buf), "Total:  %.4f J", total);
        r->draw_text(buf, x, y, sz, Rendering::palette::accent3());
        y += dy;

        std::snprintf(buf, sizeof(buf), "Drift:  %.4f%%", drift_pct);
        auto dc = drift_pct > 1.0 ? Rendering::palette::accent1()
                                  : Rendering::palette::text();
        r->draw_text(buf, x, y, sz, dc);
        y += dy;

        std::snprintf(buf, sizeof(buf), "k=%.1f  m=%.1f  f=%.3f Hz", SpringK,
                      Mass, std::sqrt(SpringK / Mass) / (2 * M_PI));
        r->draw_text(buf, x, y + 8, sz - 4, Rendering::palette::text_dim());
        y += dy;
        r->draw_text("[R] Reset  [SPACE] Kick  [H] Home", x, y + 8, sz - 4,
                     Rendering::palette::text_dim());
    }

    void render_plots(Rendering::Renderer *r) {
        int sw = r->screen_width();
        int pw = 280, ph = 90, margin = 12, gap = 6;
        int px = sw - pw - margin;

        draw_plot(
            r, px, margin, pw, ph, "Position (m)", m_pos_hist,
            [](double v) { return v; }, Rendering::palette::accent2());
        draw_plot(
            r, px, margin + ph + gap, pw, ph, "Velocity (m/s)", m_vel_hist,
            [](double v) { return v; }, Rendering::palette::accent3());
        draw_plot(
            r, px, margin + 2 * (ph + gap), pw, ph, "Total Energy (J)",
            m_energy_hist, [](double v) { return v; },
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

    Solver::RigidBody m_body;
    Solver::RigidBody m_anchor; // fixed, not in system
    Solver::LineConstraint m_rail;
    Solver::FixedRotationConstraint m_no_rot;
    Solver::Spring m_spring;

    std::deque<double> m_pos_hist, m_vel_hist, m_energy_hist;
    double m_time = 0;
    double m_initial_energy = 0;
};

} // namespace manifold::Demo
