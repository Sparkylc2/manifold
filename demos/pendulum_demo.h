#pragma once

#include "manifold/renderer/annotation_visuals.h"
#include "manifold/renderer/body_visuals.h"
#include "manifold/renderer/theme.h"
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

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class PendulumDemo : public DemoBase {
  public:
    static constexpr double AnchorMass = 0;
    static constexpr double AnchorInertia = 0;
    static constexpr double BobMass = 1.0;
    static constexpr double BarLen = 3.0;
    static constexpr double BarWidth = 0.1;
    static constexpr double Gravity = 9.81;
    static constexpr int SimSteps = 100;

    // torque = I * desired_delta_omega / dt
    // I = BobMass * BarLen^2 / 12 = 0.75
    // desired_delta_omega ≈ 3.0 rad/s, dt = 1/60
    // torque ≈ 0.75 * 3.0 * 60 = 135
    static constexpr double KickTorque = -135.0;

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

        m_kick.set_body(&m_bar);
        m_system.add_force_generator(&m_kick);

        m_mouse_spring.set_ks(1.0);
        m_mouse_spring.set_kd(8.0);
        m_mouse_spring.set_active(false);
        m_system.add_force_generator(&m_mouse_spring);
        m_grabbed = false;

        m_plot_angle.configure(
            "Angle (deg)", Rendering::palette::accent1(), 600, [](double v) {
                return Rendering::clip_angle_radians(0.0, v - M_PI / 2) *
                       180.0 / M_PI;
            });
        m_plot_energy.configure("Total Energy (J)",
                                Rendering::palette::accent3());
        m_plot_drift.configure("Constraint Drift (m)",
                               Rendering::palette::accent2());

        m_plot_angle.clear();
        m_plot_energy.clear();
        m_plot_drift.clear();
    }

    void process(double dt) override {
        m_system.process(dt, SimSteps);
        m_kick.disarm();

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

        Rendering::draw_angle_marker(
            r, m_anchor.p, -M_PI / 2, m_bar.theta - M_PI, BarLen / 6, 1.5,
            Rendering::palette::accent1(), Rendering::palette::accent1(),
            {.ref_line_len = BarLen / 4, .ref_line_thickness = 1.5});

        draw_body_node(r, m_anchor.p, 0.15,
                       {.fill = Rendering::palette::accent1()});

        draw_body_bar(r, m_bar.p, BarLen, BarWidth, m_bar.theta,
                      {.show_shadow = true});

        Vector2d bar_top;
        m_bar.local_to_world(Vector2d(BarLen / 2.0, 0), &bar_top);
        Rendering::draw_body_node(r, bar_top, 0.06, {.show_shadow = true});
        // r->draw_circle(bar_top.x(), bar_top.y(), 0.06,
        //                Rendering::palette::accent3());

        Vector2d bar_bot;
        m_bar.local_to_world(Vector2d(-BarLen / 2.0, 0), &bar_bot);
        draw_body_disk(r, bar_bot, 0.15, 0,
                       {.fill = Rendering::palette::accent2()});

        if (m_grabbed) {
            double mx, my;
            get_mouse_world(r, &mx, &my);
            Vector2d attach;
            m_bar.local_to_world(m_mouse_spring.local(), &attach);
            auto a3 = Rendering::palette::accent3();
            Rendering::draw_coil_spring(r, attach.x(), attach.y(), mx, my, 6,
                                        0.06, 2.0f, a3);
            r->draw_circle(attach.x(), attach.y(), 0.05, a3);
            r->draw_circle(mx, my, 0.04, a3);
        }

        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_angle, &m_plot_energy,
                                           &m_plot_drift};
        render_plots(r, plots, 280, 90);
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

        double mx, my;
        get_mouse_world(r, &mx, &my);
        Vector2d mouse(mx, my);

        if (r->is_mouse_button_pressed(Rendering::mouse::Left)) {
            Vector2d local;
            m_bar.world_to_local(mouse, &local);
            if (std::abs(local.x()) <= BarLen / 2 + 0.15 &&
                std::abs(local.y()) <= 0.3) {
                m_grabbed = true;
                m_mouse_spring.set_body(&m_bar);
                m_mouse_spring.set_local(local);
                m_mouse_spring.set_target(mouse);
                m_mouse_spring.set_active(true);
            }
        }
        if (m_grabbed && r->is_mouse_button_down(Rendering::mouse::Left))
            m_mouse_spring.set_target(mouse);
        if (m_grabbed && !r->is_mouse_button_down(Rendering::mouse::Left)) {
            m_grabbed = false;
            m_mouse_spring.set_active(false);
        }
    }

  private:
    void get_mouse_world(Rendering::Renderer *r, double *wx, double *wy) {
        int mx, my;
        r->get_mouse_pos(&mx, &my);
        r->screen_to_world(mx, my, wx, wy);
    }

    void render_hud(Rendering::Renderer *r) {
        double ke = m_anchor.energy() + m_bar.energy();
        double pe = m_bar.m * Gravity * m_bar.p.y();

        Vector2d bar_top;
        m_bar.local_to_world(Vector2d(BarLen / 2.0, 0), &bar_top);
        double drift = (bar_top - m_anchor.p).norm();

        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("PENDULUM", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Angle:  %.2f deg",
                 Rendering::clip_angle_radians(0.0, m_bar.theta - M_PI / 2) *
                     180.0 / M_PI);
        hud.line(Rendering::palette::text(), "KE:     %.4f J", ke);
        hud.line(Rendering::palette::text(), "PE:     %.4f J", pe);
        hud.line(Rendering::palette::accent3(), "Total:  %.4f J", ke + pe);
        hud.line(drift > 0.01 ? Rendering::palette::accent1()
                              : Rendering::palette::text(),
                 "Drift:  %.6f m", drift);
        hud.separator();
        hud.small_text("[LMB] Drag  [SPACE] Kick (+Shift reverses)",
                       Rendering::palette::text_dim());
        hud.small_text("[R] Reset  [H] Home", Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_anchor, m_bar;
    Solver::FixedPositionConstraint m_pin;
    Solver::LinkConstraint m_link;
    Solver::UniformGravityForceGenerator m_gravity;
    Solver::ImpulseForceGenerator m_kick;
    Solver::MouseSpringForceGenerator m_mouse_spring;
    bool m_grabbed = false;

    PlotWidget m_plot_energy, m_plot_angle, m_plot_drift;
};

} // namespace manifold::Demo
