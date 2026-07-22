#pragma once

#include "manifold/solver/forces/impulse.h"
#include "manifold/solver/forces/uniform_gravity.h"
#include <manifold/control/pid.h>
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

class CartPendulumDemo : public DemoBase {
  public:
    static constexpr double CartMass = 5.0;
    static constexpr double CartWidth = 1.2;
    static constexpr double CartHeight = 0.5;
    static constexpr double PendulumMass = 1.0;
    static constexpr double PendulumLen = 3.0;
    static constexpr double PendulumWidth = 0.12;
    static constexpr double Gravity = 9.81;
    static constexpr int SimSteps = 100;

    static constexpr double KickForce = -135.0;

    enum class TunerTarget { Angle, Position };
    enum class TunerParam { Kp, Ki, Kd };

    const char *name() const override { return "Cart-Pendulum"; }
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

        m_kick.set_body(&m_pendulum);
        m_system.add_force_generator(&m_kick);

        apply_gains();

        m_plot_angle.configure("Angle (deg)", Rendering::palette::accent2(),
                               600, [](double v) { return v * 180.0 / M_PI; });
        m_plot_cart.configure("Cart X (m)", Rendering::palette::accent3());
        m_plot_force.configure("Force (N)", Rendering::palette::accent1());
        m_plot_energy.configure("Energy (J)", Rendering::palette::foreground());

        m_plot_angle.clear();
        m_plot_cart.clear();
        m_plot_force.clear();
        m_plot_energy.clear();
        m_last_force = 0;

        clear_overlays();
        Rendering::register_constraint_overlays(*this, m_system);
    }

    void process(double dt) override {
        double angle_err = angle_error(m_pendulum.theta, M_PI / 2.0);
        double pos_err = 0.0 + m_cart.p.x();

        double f_angle = m_pid_angle.update(angle_err, dt);
        double f_pos = m_pid_pos.update(pos_err, dt);
        double force = f_angle + f_pos;

        m_control_force.set_force(Vector2d(force, 0));
        m_last_force = force;

        m_system.process(dt, SimSteps);
        m_kick.disarm();

        double ke = m_cart.energy() + m_pendulum.energy();
        double pe = m_pendulum.m * Gravity * m_pendulum.p.y() +
                    m_cart.m * Gravity * m_cart.p.y();

        m_plot_angle.push(m_pendulum.theta);
        m_plot_cart.push(m_cart.p.x());
        m_plot_force.push(force);
        m_plot_energy.push(ke + pe);
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        r->draw_line(-30, 0, 30, 0, 3.0f, Rendering::palette::grid_axis());
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

        std::vector<PlotWidget *> plots = {&m_plot_angle, &m_plot_cart,
                                           &m_plot_force, &m_plot_energy};
        render_plots(r, plots);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space)) {
            bool rev = r->is_key_down(Rendering::keys::LeftShift) ||
                       r->is_key_down(Rendering::keys::RightShift);
            const Vector2d disp = m_pendulum.p - m_cart.p;
            const double theta = std::atan2(disp.y(), disp.x());
            Vector2d force_n = {std::cos(theta), std::sin(theta)};
            m_kick.arm_force(rev ? -KickForce * force_n : KickForce * force_n);
        }

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

        if (r->is_key_down(Rendering::keys::D))
            adjust_gain(step);
        if (r->is_key_down(Rendering::keys::A))
            adjust_gain(-step);
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
        Rendering::HUDPanel hud(r, 12, 12);
        double ke = m_cart.energy() + m_pendulum.energy();
        double pe = m_pendulum.m * Gravity * m_pendulum.p.y();

        hud.title("CART-PENDULUM PID", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Angle:  %.2f deg",
                 m_pendulum.theta * 180.0 / M_PI);
        hud.line(Rendering::palette::text(), "Cart X: %.3f m", m_cart.p.x());
        hud.line(Rendering::palette::text(), "Force:  %.1f N", m_last_force);
        hud.line(Rendering::palette::accent3(), "Energy: %.2f J", ke + pe);
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

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_cart, m_pendulum;
    Solver::LineConstraint m_rail;
    Solver::FixedRotationConstraint m_cart_rot;
    Solver::LinkConstraint m_pivot;
    Solver::UniformGravityForceGenerator m_gravity;
    Solver::DirectForceGenerator m_control_force;
    Solver::ImpulseForceGenerator m_kick;

    Control::PIDController m_pid_angle;
    double m_kp_a = 150.0, m_ki_a = 2.0, m_kd_a = 40.0;

    Control::PIDController m_pid_pos;
    double m_kp_p = 3.0, m_ki_p = 0.2, m_kd_p = 5.0;

    double m_last_force = 0;

    TunerTarget m_tuner_target = TunerTarget::Angle;
    TunerParam m_tuner_param = TunerParam::Kp;

    PlotWidget m_plot_angle, m_plot_cart, m_plot_force, m_plot_energy;
};

} // namespace manifold::Demo
