#pragma once

#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/utilities.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/impulse.h>
#include <manifold/solver/forces/mouse_spring.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class DoublePendulumDemo : public DemoBase {
  public:
    static constexpr double L1 = 2.5;
    static constexpr double L2 = 2.0;
    static constexpr double BarWidth = 0.1;
    static constexpr double M1 = 3.0;
    static constexpr double M2 = 2.0;
    static constexpr double Gravity = 9.81;
    static constexpr int SimSteps = 100;

    // displacement from the downward vertical (theta = pi/2 hangs straight down)
    static constexpr double InitAngle1 = 0.6;
    static constexpr double InitAngle2 = 0.4;
    static constexpr double KickTorque = 120.0;

    const char *name() const override { return "Double Pendulum"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return -2.0; }
    double default_cam_zoom() const override { return 50.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_anchor.reset();
        m_anchor.m = 0;
        m_anchor.I = 0;
        m_anchor.p = Vector2d(0, 0);
        m_system.add_body(&m_anchor);

        // pinned end (+L/2) sits at the origin, so the bar hangs from it
        double theta1 = M_PI / 2.0 + InitAngle1;
        m_bar1.reset();
        m_bar1.m = M1;
        m_bar1.I = M1 * L1 * L1 / 12.0;
        m_bar1.theta = theta1;
        m_bar1.p =
            Vector2d(-std::cos(theta1) * L1 / 2.0, -std::sin(theta1) * L1 / 2.0);
        m_system.add_body(&m_bar1);

        Vector2d joint1;
        m_bar1.local_to_world(Vector2d(-L1 / 2.0, 0), &joint1);

        double theta2 = M_PI / 2.0 + InitAngle2;
        m_bar2.reset();
        m_bar2.m = M2;
        m_bar2.I = M2 * L2 * L2 / 12.0;
        m_bar2.theta = theta2;
        m_bar2.p = Vector2d(joint1.x() - std::cos(theta2) * L2 / 2.0,
                            joint1.y() - std::sin(theta2) * L2 / 2.0);
        m_system.add_body(&m_bar2);

        m_pin.set_body(&m_anchor);
        m_pin.set_world_position(Vector2d(0, 0));
        m_pin.set_local_position(Vector2d(0, 0));
        m_pin.set_ks(100.0);
        m_pin.set_kd(10.0);
        m_system.add_constraint(&m_pin);

        m_link1.set_bodies(&m_anchor, &m_bar1);
        m_link1.set_local_pos1(Vector2d(0, 0));
        m_link1.set_local_pos2(Vector2d(L1 / 2.0, 0));
        m_link1.set_ks(100.0);
        m_link1.set_kd(10.0);
        m_system.add_constraint(&m_link1);

        m_link2.set_bodies(&m_bar1, &m_bar2);
        m_link2.set_local_pos1(Vector2d(-L1 / 2.0, 0));
        m_link2.set_local_pos2(Vector2d(L2 / 2.0, 0));
        m_link2.set_ks(100.0);
        m_link2.set_kd(10.0);
        m_system.add_constraint(&m_link2);

        m_gravity.set_gravity(Gravity);
        m_system.add_force_generator(&m_gravity);

        m_kick.set_body(&m_bar2);
        m_system.add_force_generator(&m_kick);

        m_mouse_spring.set_ks(80.0);
        m_mouse_spring.set_kd(8.0);
        m_mouse_spring.set_active(false);
        m_system.add_force_generator(&m_mouse_spring);

        auto from_vertical = [](double v) {
            return Rendering::clip_angle_radians(0.0, v - M_PI / 2.0) * 180.0 /
                   M_PI;
        };
        m_plot_theta1.configure("θ₁ (deg)", Rendering::palette::accent2(), 600,
                                from_vertical);
        m_plot_theta2.configure("θ₂ (deg)", Rendering::palette::accent3(), 600,
                                from_vertical);
        m_plot_energy.configure("Total Energy (J)",
                                Rendering::palette::accent1());
        m_phase.configure("θ₁ (deg)", "θ₂ (deg)",
                          Rendering::palette::accent4());

        m_plot_theta1.clear();
        m_plot_theta2.clear();
        m_plot_energy.clear();
        m_phase.clear();
        m_trail.clear();
        m_grabbed_body = nullptr;
    }

    void process(double dt) override {
        m_system.process(dt, SimSteps);
        m_kick.disarm();

        m_plot_theta1.push(m_bar1.theta);
        m_plot_theta2.push(m_bar2.theta);

        double ke = m_bar1.energy() + m_bar2.energy();
        double pe = M1 * Gravity * m_bar1.p.y() + M2 * Gravity * m_bar2.p.y();
        m_plot_energy.push(ke + pe);

        m_phase.push(angle_deg(m_bar1.theta), angle_deg(m_bar2.theta));

        Vector2d tip;
        m_bar2.local_to_world(Vector2d(-L2 / 2.0, 0), &tip);
        m_trail.push_back(tip);
        if ((int)m_trail.size() > 800)
            m_trail.erase(m_trail.begin());
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        auto dim = Rendering::palette::text_dim();
        auto fg = Rendering::palette::foreground();
        auto a1 = Rendering::palette::accent1();
        auto a2 = Rendering::palette::accent2();
        auto a3 = Rendering::palette::accent3();

        // ---- trail of the tip ----
        for (int i = 1; i < (int)m_trail.size(); ++i) {
            double alpha = (double)i / m_trail.size();
            auto tc = Rendering::Color::rgba(
                a1.r, a1.g, a1.b, (unsigned char)(200 * alpha));
            r->draw_line(m_trail[i - 1].x(), m_trail[i - 1].y(),
                         m_trail[i].x(), m_trail[i].y(), 1.5f, tc);
        }

        Vector2d joint;
        m_bar1.local_to_world(Vector2d(-L1 / 2.0, 0), &joint);
        Vector2d tip;
        m_bar2.local_to_world(Vector2d(-L2 / 2.0, 0), &tip);

        // ---- angle annotations: both measured from the downward vertical ----
        Rendering::draw_angle_marker(
            r, 0.0, 0.0, -M_PI / 2.0, m_bar1.theta - M_PI, L1 * 0.18, 1.5f, a2,
            dim, {.ref_line_len = L1 * 0.32, .ref_line_thickness = 1.5});
        Rendering::draw_angle_marker(
            r, joint.x(), joint.y(), -M_PI / 2.0, m_bar2.theta - M_PI, L2 * 0.18,
            1.5f, a3, dim, {.ref_line_len = L2 * 0.32, .ref_line_thickness = 1.5});

        // ---- scene ----
        draw_body_node(r, Vector2d(0, 0), 0.15, {.fill = a2});
        draw_body_bar(r, m_bar1.p, L1, BarWidth, m_bar1.theta,
                      {.fill = a2, .show_shadow = true});
        draw_body_bar(r, m_bar2.p, L2, BarWidth, m_bar2.theta,
                      {.fill = a3, .show_shadow = true});
        draw_body_node(r, Vector2d(0, 0), 0.06, {.show_shadow = false});
        draw_body_node(r, joint, 0.09, {.show_shadow = true});
        draw_body_disk(r, tip, 0.15, m_bar2.theta,
                       {.fill = Rendering::palette::accent4()});

        // ---- mouse spring ----
        if (m_grabbed_body) {
            double mx, my;
            get_mouse_world(r, &mx, &my);
            Vector2d attach;
            m_grabbed_body->local_to_world(m_mouse_spring.local(), &attach);
            Rendering::draw_coil_spring(r, attach.x(), attach.y(), mx, my, 6,
                                        0.06, 2.0f, fg);
            r->draw_circle(attach.x(), attach.y(), 0.05, a1);
            r->draw_circle(mx, my, 0.04, a1);
        }

        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_theta1, &m_plot_theta2,
                                           &m_plot_energy};
        render_plots(r, plots, 280, 80);

        if (!portrait_mode()) {
            int px = r->screen_width() - 280 - 12;
            int py = 12 + 3 * (80 + 6) + 14;
            m_phase.render(r, px, py, 280, 220);
        }
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space)) {
            bool rev = r->is_key_down(Rendering::keys::LeftShift) ||
                       r->is_key_down(Rendering::keys::RightShift);
            m_kick.arm_torque(rev ? -KickTorque : KickTorque);
        }
        if (r->is_key_pressed(Rendering::keys::C))
            m_trail.clear();

        double mx, my;
        get_mouse_world(r, &mx, &my);
        Vector2d mouse(mx, my);

        if (r->is_mouse_button_pressed(Rendering::mouse::Left)) {
            Vector2d local;
            m_grabbed_body = pick_body(mouse, &local);
            if (m_grabbed_body) {
                m_mouse_spring.set_body(m_grabbed_body);
                m_mouse_spring.set_local(local);
                m_mouse_spring.set_target(mouse);
                m_mouse_spring.set_active(true);
            }
        }
        if (m_grabbed_body && r->is_mouse_button_down(Rendering::mouse::Left))
            m_mouse_spring.set_target(mouse);
        if (m_grabbed_body &&
            !r->is_mouse_button_down(Rendering::mouse::Left)) {
            m_grabbed_body = nullptr;
            m_mouse_spring.set_active(false);
        }
    }

  private:
    static double angle_deg(double theta) {
        return Rendering::clip_angle_radians(0.0, theta - M_PI / 2.0) * 180.0 /
               M_PI;
    }

    // grab whichever bar the click lands on, attaching at the exact local point
    Solver::RigidBody *pick_body(const Vector2d &mouse, Vector2d *local_out) {
        struct Cand {
            Solver::RigidBody *b;
            double half_len;
        };
        Cand cands[] = {{&m_bar1, L1 / 2.0}, {&m_bar2, L2 / 2.0}};

        Solver::RigidBody *best = nullptr;
        double best_off = 1e9;
        for (auto &c : cands) {
            Vector2d local;
            c.b->world_to_local(mouse, &local);
            if (std::abs(local.x()) <= c.half_len + 0.15 &&
                std::abs(local.y()) <= 0.25 && std::abs(local.y()) < best_off) {
                best_off = std::abs(local.y());
                best = c.b;
                *local_out = local;
            }
        }
        return best;
    }

    void get_mouse_world(Rendering::Renderer *r, double *wx, double *wy) {
        int mx, my;
        r->get_mouse_pos(&mx, &my);
        r->screen_to_world(mx, my, wx, wy);
    }

    void render_hud(Rendering::Renderer *r) {
        double ke = m_bar1.energy() + m_bar2.energy();
        double pe = M1 * Gravity * m_bar1.p.y() + M2 * Gravity * m_bar2.p.y();

        Vector2d tip;
        m_bar2.local_to_world(Vector2d(-L2 / 2.0, 0), &tip);

        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("DOUBLE PENDULUM", Rendering::palette::accent2());
        hud.line(Rendering::palette::accent2(), "θ₁:     %.2f deg",
                 angle_deg(m_bar1.theta));
        hud.line(Rendering::palette::accent3(), "θ₂:     %.2f deg",
                 angle_deg(m_bar2.theta));
        hud.line(Rendering::palette::text(), "Tip:    (%.2f, %.2f)", tip.x(),
                 tip.y());
        hud.line(Rendering::palette::text(), "KE:     %.3f J", ke);
        hud.line(Rendering::palette::text(), "PE:     %.3f J", pe);
        hud.line(Rendering::palette::accent1(), "Total:  %.3f J", ke + pe);
        hud.separator();
        hud.small_text("[LMB] Drag  [SPACE] Kick (+Shift reverses)",
                       Rendering::palette::text_dim());
        hud.small_text("[C] Clear trail  [R] Reset  [H] Home",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_anchor, m_bar1, m_bar2;
    Solver::FixedPositionConstraint m_pin;
    Solver::LinkConstraint m_link1, m_link2;
    Solver::UniformGravityForceGenerator m_gravity;
    Solver::ImpulseForceGenerator m_kick;
    Solver::MouseSpringForceGenerator m_mouse_spring;

    Solver::RigidBody *m_grabbed_body = nullptr;
    std::vector<Vector2d> m_trail;

    PlotWidget m_plot_theta1, m_plot_theta2, m_plot_energy;
    Rendering::PhasePlot m_phase;
};

} // namespace manifold::Demo
