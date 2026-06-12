#pragma once

#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class PendulumDemo : public DemoBase {
  public:
    static constexpr double AnchorMass = 1e6;
    static constexpr double AnchorInertia = 1e6;
    static constexpr double BobMass = 1.0;
    static constexpr double BarLen = 3.0;
    static constexpr double BarWidth = 0.1;
    static constexpr double Gravity = 9.81;
    static constexpr int SimSteps = 100;

    const char *name() const override { return "Pendulum"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return -1.0; }
    double default_cam_zoom() const override { return 60.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

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

        m_plot_angle.configure("Angle (deg)", Rendering::palette::accent2(),
                               600, [](double v) { return v * 180.0 / M_PI; });
        m_plot_energy.configure("Total Energy (J)",
                                Rendering::palette::accent3());
        m_plot_drift.configure("Constraint Drift (m)",
                               Rendering::palette::accent1());

        m_plot_angle.clear();
        m_plot_energy.clear();
        m_plot_drift.clear();
    }

    void process(double dt) override {
        m_system.process(dt, SimSteps);

        double ke = m_anchor.energy() + m_bar.energy();
        double pe = m_bar.m * Gravity * m_bar.p.y();

        Vector2d bar_top;
        m_bar.local_to_world(Vector2d(BarLen / 2.0, 0), &bar_top);
        double drift = (bar_top - m_anchor.p).norm();

        m_plot_energy.push(ke + pe);
        m_plot_angle.push(m_bar.theta);
        m_plot_drift.push(drift);
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        r->draw_circle(m_anchor.p.x(), m_anchor.p.y(), 0.15,
                       Rendering::palette::accent1());

        r->draw_bar(m_bar.p.x(), m_bar.p.y(), m_bar.theta, BarLen, BarWidth,
                    Rendering::palette::accent2(),
                    Rendering::palette::shadow());

        Vector2d bar_top;
        m_bar.local_to_world(Vector2d(BarLen / 2.0, 0), &bar_top);
        r->draw_circle(bar_top.x(), bar_top.y(), 0.06,
                       Rendering::palette::accent3());

        Vector2d bar_bot;
        m_bar.local_to_world(Vector2d(-BarLen / 2.0, 0), &bar_bot);
        r->draw_circle(bar_bot.x(), bar_bot.y(), 0.08,
                       Rendering::palette::foreground());

        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_angle, &m_plot_energy,
                                           &m_plot_drift};
        render_plots(r, plots, 280, 90);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space))
            m_bar.v_theta += 3.0;
    }

  private:
    void render_hud(Rendering::Renderer *r) {
        double ke = m_anchor.energy() + m_bar.energy();
        double pe = m_bar.m * Gravity * m_bar.p.y();

        Vector2d bar_top;
        m_bar.local_to_world(Vector2d(BarLen / 2.0, 0), &bar_top);
        double drift = (bar_top - m_anchor.p).norm();

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("PENDULUM", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Angle:  %.2f deg",
                 m_bar.theta * 180.0 / M_PI);
        hud.line(Rendering::palette::text(), "KE:     %.4f J", ke);
        hud.line(Rendering::palette::text(), "PE:     %.4f J", pe);
        hud.line(Rendering::palette::accent3(), "Total:  %.4f J", ke + pe);
        hud.line(drift > 0.01 ? Rendering::palette::accent1()
                              : Rendering::palette::text(),
                 "Drift:  %.6f m", drift);
        hud.separator();
        hud.small_text("[R] Reset  [SPACE] Kick  [H] Home",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_anchor, m_bar;
    Solver::FixedPositionConstraint m_pin;
    Solver::LinkConstraint m_link;
    Solver::GravityForceGenerator m_gravity;

    PlotWidget m_plot_energy, m_plot_angle, m_plot_drift;
};

} // namespace manifold::Demo
