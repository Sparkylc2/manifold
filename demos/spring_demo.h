#pragma once

#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class SpringDemo : public DemoBase {
  public:
    static constexpr double Mass = 2.0;
    static constexpr double BoxSize = 0.6;
    static constexpr double SpringK = 20.0;
    static constexpr double SpringDamp = 0.0;
    static constexpr double StartX = 2.0;
    static constexpr int SimSteps = 100;

    const char *name() const override { return "Spring"; }
    double default_cam_x() const override { return 0.5; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 70.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_body.reset();
        m_body.m = Mass;
        m_body.I = Mass * BoxSize * BoxSize / 6.0;
        m_body.p = Vector2d(StartX, 0);
        m_system.add_body(&m_body);

        m_rail.set_body(&m_body);
        m_rail.set_line(Vector2d(0, 0), Vector2d(1, 0));
        m_rail.set_local_pos(Vector2d(0, 0));
        m_rail.set_ks(100.0);
        m_rail.set_kd(10.0);
        m_system.add_constraint(&m_rail);

        m_no_rot.set_body(&m_body);
        m_no_rot.set_angle(0);
        m_no_rot.set_ks(100.0);
        m_no_rot.set_kd(10.0);
        m_system.add_constraint(&m_no_rot);

        m_anchor.reset();
        m_anchor.m = 1.0;
        m_anchor.I = 1.0;
        m_anchor.p = Vector2d(0, 0);

        m_spring.set_bodies(&m_anchor, &m_body);
        m_spring.set_local_pos1(Vector2d(0, 0));
        m_spring.set_local_pos2(Vector2d(0, 0));
        m_spring.set_rest_length(0.0);
        m_spring.set_ks(SpringK);
        m_spring.set_kd(SpringDamp);
        m_system.add_force_generator(&m_spring);

        m_plot_pos.configure("Position (m)", Rendering::palette::accent2());
        m_plot_vel.configure("Velocity (m/s)", Rendering::palette::accent3());
        m_plot_energy.configure("Total Energy (J)",
                                Rendering::palette::accent1());
        m_plot_pos.clear();
        m_plot_vel.clear();
        m_plot_energy.clear();
        m_initial_energy = 0.5 * SpringK * StartX * StartX;
    }

    void process(double dt) override {
        m_system.process(dt, SimSteps);

        double pe = m_spring.energy();
        double ke = m_body.energy();

        m_plot_pos.push(m_body.p.x());
        m_plot_vel.push(m_body.v.x());
        m_plot_energy.push(pe + ke);
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        auto dim = Rendering::palette::text_dim();
        auto ann = Rendering::Color::rgba(100, 160, 200, 180);
        auto disp_col = Rendering::palette::accent3();
        auto vel_col = Rendering::palette::accent2();
        auto force_col = Rendering::palette::accent1();

        // ---- annotations (behind the scene) ----

        // equilibrium reference: dashed vertical at x=0
        Rendering::draw_dashed_line(r, 0, -2, 0, 2, 1.0f, dim, 0.12, 0.08);
        Rendering::draw_reference_cross(r, 0, 0, 0.08, 1.5f, dim);

        // equilibrium label
        int sx_eq, sy_eq;
        r->world_to_screen(0, 1.8, &sx_eq, &sy_eq);
        r->draw_text("x = 0", sx_eq - 14, sy_eq - 8, 12, dim);

        // displacement arrow from equilibrium to mass center
        double x = m_body.p.x();
        if (std::abs(x) > 0.05) {
            Rendering::draw_displacement(r, 0, 0, x, 0, "x = %.3f m", x, 2.0f,
                                         disp_col, -0.5);
        }

        // velocity arrow on the mass
        double vx = m_body.v.x();
        if (std::abs(vx) > 0.05) {
            Rendering::draw_velocity_arrow(r, x, BoxSize * 0.5 + 0.15, vx, 0,
                                           0.08, 2.0f, vel_col);
            // label
            int sx_v, sy_v;
            r->world_to_screen(x + vx * 0.04, BoxSize * 0.5 + 0.35, &sx_v,
                               &sy_v);
            char vbuf[32];
            std::snprintf(vbuf, sizeof(vbuf), "v = %.2f", vx);
            r->draw_text(vbuf, sx_v - 20, sy_v - 8, 12, vel_col);
        }

        // spring force arrow on the mass
        double spring_force = -SpringK * x; // F = -kx
        if (std::abs(spring_force) > 0.5) {
            Rendering::draw_force_arrow(r, x, -(BoxSize * 0.5 + 0.15),
                                        spring_force, 0, 0.003, 2.5f,
                                        force_col);
            int sx_f, sy_f;
            r->world_to_screen(x + spring_force * 0.0015,
                               -(BoxSize * 0.5 + 0.35), &sx_f, &sy_f);
            char fbuf[32];
            std::snprintf(fbuf, sizeof(fbuf), "F = %.1f N", spring_force);
            r->draw_text(fbuf, sx_f - 24, sy_f - 8, 12, force_col);
        }

        // ---- scene ----

        // rail
        r->draw_line(-6, 0, 6, 0, 1.5f, dim);

        // wall at anchor
        r->draw_line(-0.05, -0.6, -0.05, 0.6, 2.5f,
                     Rendering::palette::foreground());
        for (int i = -3; i <= 3; ++i) {
            double hy = i * 0.15;
            r->draw_line(-0.05, hy, -0.25, hy - 0.12, 1.5f, dim);
        }

        // spring coil
        draw_spring_visual(r, 0, 0, x, 0, 10, 0.2);

        // mass block
        r->draw_bar(x, 0, 0, BoxSize, BoxSize, Rendering::palette::accent2(),
                    Rendering::palette::shadow());

        // center dot on mass
        r->draw_circle(x, 0, 0.03, Rendering::palette::background());

        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_pos, &m_plot_vel,
                                           &m_plot_energy};
        render_plots(r, plots, 280, 90);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space))
            m_body.v.x() += 3.0;
    }

  private:
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
        double pe = m_spring.energy();
        double ke = m_body.energy();
        double total = pe + ke;
        double drift_pct =
            (m_initial_energy > 0)
                ? std::abs(total - m_initial_energy) / m_initial_energy * 100.0
                : 0.0;

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("SPRING OSCILLATOR", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Pos:    %.4f m", m_body.p.x());
        hud.line(Rendering::palette::text(), "Vel:    %.4f m/s", m_body.v.x());
        hud.line(Rendering::palette::text(), "KE:     %.4f J", ke);
        hud.line(Rendering::palette::text(), "PE:     %.4f J", pe);
        hud.line(Rendering::palette::accent3(), "Total:  %.4f J", total);
        hud.line(drift_pct > 1.0 ? Rendering::palette::accent1()
                                 : Rendering::palette::text(),
                 "Drift:  %.4f%%", drift_pct);
        hud.separator();
        hud.small_text("k=20.0  m=2.0  f=0.503 Hz",
                       Rendering::palette::text_dim());
        hud.small_text("[R] Reset  [SPACE] Kick  [H] Home",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_body;
    Solver::RigidBody m_anchor;
    Solver::LineConstraint m_rail;
    Solver::FixedRotationConstraint m_no_rot;
    Solver::Spring m_spring;

    PlotWidget m_plot_pos, m_plot_vel, m_plot_energy;
    double m_initial_energy = 0;
};

} // namespace manifold::Demo
