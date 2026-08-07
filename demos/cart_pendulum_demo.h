#pragma once

#include "manifold/renderer/theme.h"
#include "manifold/solver/forces/impulse.h"
#include "manifold/solver/forces/uniform_gravity.h"
#include <cstdio>
#include <iostream>
#include <manifold/control/pid.h>
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

class CartPendulumDemo : public DemoBase {
  public:
    static constexpr double CartMass = 5.0;
    static constexpr double CartWidth = 1.2;
    static constexpr double CartHeight = 0.34;
    static constexpr double PendulumMass = 1.0;
    static constexpr double PendulumLen = 3.0;
    static constexpr double PendulumWidth = 0.085;
    static constexpr double Gravity = 9.81;
    static constexpr int SimSteps = 100;

    static constexpr double KickForce = -135.0;
    static constexpr double WheelR = 0.13;
    // the grid's x-axis should read as the ground the wheels roll on, so it
    // sits a wheel below the cart body rather than through its centre of mass
    static constexpr double GroundY =
        -(CartHeight * 0.5 + 2.0 * WheelR + 0.035);

    // Procedural nudge. Specified as an *impulse* (N s) over a duration,
    // because that is what actually moves the system: the old one-frame
    // arm/disarm applied its force for a single dt, so 110 N was 110/240 =
    // 0.46 N s and raising the number barely moved the needle. A half-sine
    // push over ~0.3 s delivers two orders more for a smaller peak force, and
    // gives the controller something continuous to fight.
    void set_auto_kick(double first_at, double period, double impulse_ns,
                       double duration = 0.30) {
        m_kick_at = first_at;
        m_kick_period = period;
        m_kick_dur = std::max(1e-3, duration);
        m_kick_amp = impulse_ns / (m_kick_dur * 2.0 / M_PI);
    }

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
        m_force_shown = 0.0;
        m_force_vel = 0.0;
        m_force_peak = 0.0;
        m_kick_until = -1.0;
        m_t = 0.0;

        clear_overlays();
        Rendering::register_constraint_overlays(*this, m_system);
    }

    void process(double dt) override {
        m_t += dt;
        if (m_kick_at > 0.0 && m_t >= m_kick_at) {
            m_kick_until = m_t + m_kick_dur;
            m_kick_at = m_kick_period > 0.0 ? m_t + m_kick_period : 0.0;
        }
        if (m_t < m_kick_until) {
            // half-sine, so it eases in and out rather than stepping
            const double u = 1.0 - (m_kick_until - m_t) / m_kick_dur;
            m_kick.arm_force(Vector2d(
                m_kick_amp * std::sin(M_PI * std::clamp(u, 0.0, 1.0)), 0.0));
        }

        double angle_err = angle_error(m_pendulum.theta, M_PI / 2.0);
        double pos_err = 0.0 + m_cart.p.x();

        double f_angle = m_pid_angle.update(angle_err, dt);
        double f_pos = m_pid_pos.update(pos_err, dt);
        double force = f_angle + f_pos;

        m_control_force.set_force(Vector2d(force, 0));
        m_last_force = force;

        // second-order critically damped follower: the arrow eases up to the
        // command and eases back, instead of tracking its chatter. an EMA
        // lagged but still stepped; this ramps
        const double wn = 14.0;
        const double acc =
            wn * wn * (m_last_force - m_force_shown) - 2.0 * wn * m_force_vel;
        m_force_vel += acc * dt;
        m_force_shown += m_force_vel * dt;

        // slowly-decaying peak, so the arrow spans its full length range
        // whatever the controller's actual force scale turns out to be
        m_force_peak = std::max(std::abs(m_force_shown),
                                m_force_peak * std::exp(-0.25 * dt));

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
        render_cell(r);

        render_hud(r);
        render_tuner(r);

        std::vector<PlotWidget *> plots = {&m_plot_angle, &m_plot_cart,
                                           &m_plot_force, &m_plot_energy};
        render_plots(r, plots);
    }

    void render_cell(Rendering::Renderer *r) override {
        const auto fg = Rendering::palette::foreground();
        const auto dim = Rendering::palette::text_dim();
        const auto a1 = Rendering::palette::accent1();
        const auto a2 = Rendering::palette::accent2();
        const auto a3 = Rendering::palette::accent3();
        const auto blue = Rendering::palette::accent4();

        // draw_reference(r, a3);

        // wheels, dropped clear of the body's outline thickness
        const double wy = -CartHeight * 0.5 - WheelR - 0.035;
        // rolling without slipping: moving +x spins the wheel clockwise, which
        // is negative in a y-up frame. draw_disk draws its spoke at -theta, so
        // it is handed -a to line up with the dot
        const double a = -m_cart.p.x() / WheelR;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            const double wx = m_cart.p.x() + sgn * CartWidth * 0.34;
            Rendering::draw_body_disk(r, wx, wy, WheelR, -a,
                                      {.fill = dim, .show_shadow = true});
            r->draw_circle(wx + 0.55 * WheelR * std::cos(a),
                           wy + 0.55 * WheelR * std::sin(a), 0.028, fg);
        }

        Rendering::draw_body_block_circular(r, m_cart.p, CartWidth, CartHeight,
                                            0.0, {.show_shadow = true});

        Rendering::draw_body_bar(
            r, m_pendulum.p, PendulumLen, PendulumWidth, m_pendulum.theta,
            {.fill = a2, .show_center = false, .show_shadow = true});

        Vector2d piv;
        m_cart.local_to_world(Vector2d(0, 0), &piv);
        Vector2d tip;
        m_pendulum.local_to_world(Vector2d(PendulumLen / 2.0, 0), &tip);

        Rendering::draw_body_disk(r, tip, 0.13, m_pendulum.theta,
                                  {.fill = a3, .show_shadow = true});

        Rendering::draw_body_disk(r, piv, 0.085, 0,
                                  {.show_center = false,
                                   .fill = Rendering::palette::accent2(),
                                   .show_shadow = true});

        // Rendering::draw_pin_joint(
        //     r, piv, {.radius = 0.085, .fill =
        //     Rendering::palette::accent2()});

        if (std::abs(m_cart.p.x()) > 0.02)
            Rendering::draw_displacement(r, -0.04, GroundY, m_cart.p.x(),
                                         GroundY, "x = %.2f m", m_cart.p.x(),
                                         2.0f, a1, 0.0, {.offset = -0.62});

        // control effort. the raw command chatters, so this is an EMA — an
        // arrow that flickers reads as noise rather than as a quantity — and
        // it is length-proportional with a floor so the head stays legible
        // normalised against the running peak rather than a fixed newton
        // reference: a hard clamp floor is what made this look like one fixed
        // length, because the real command sits well under any guess at range
        const double ref = std::max(m_force_peak, 1.0);
        const double u = std::clamp(std::abs(m_force_shown) / ref, 0.0, 1.0);
        if (u > 0.02) {
            const double len = CartWidth * (0.09 + 1.15 * std::pow(u, 0.7));
            const double dir = m_force_shown > 0 ? 1.0 : -1.0;
            // starts at the cart face the force pushes toward, so it reads as
            // a load on the body rather than a floating annotation
            const double x0 = m_cart.p.x() + dir * (CartWidth * 0.5 + 0.145);
            const double ay = m_cart.p.y();
            draw_vector(r, x0, ay, dir * len, 0.0, 3.0f, blue);

            Rendering::LayerScope txt(r, Rendering::Layer::Text);
            char buf[24];
            std::snprintf(buf, sizeof buf, "%.0f N", m_force_shown);
            int sx, sy;
            r->world_to_screen(x0 + dir * len, ay, &sx, &sy);
            const int tw = r->measure_text(buf, 11);
            r->draw_text(buf, dir > 0 ? sx + 7 : sx - tw - 7, sy - 6, 11, blue);
        }
    }

    // Renderer::draw_arrow puts a fixed 12 *screen* px head on the end, so at
    // this scale a short vector was head and nothing else. This keeps the head
    // proportional to the vector in world units, so length reads as magnitude.
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
        const double theta = std::atan2(by, bx);
        const double length = std::sqrt(bx * bx + by * by);
        r->draw_line(x0, y0, bx, by, thickness, c);
        r->draw_triangle(x0 + dx, y0 + dy, bx - uy * hw, by + ux * hw,
                         bx + uy * hw, by - ux * hw, c);
    }

    // dashed setpoint marker, label upright and set off to the left so it
    // reads against the grid rather than on top of the pendulum
    void draw_reference(Rendering::Renderer *r, Rendering::Color c) const {
        // hangs below the origin, a little past the displacement annotation's
        // own extension lines (offset -0.62) so it reads as the datum they
        // are measured from
        const double y0 = GroundY, y1 = GroundY - 0.80;
        const double dash = 0.09, gap = 0.075;
        for (double y = y0; y > y1; y -= dash + gap)
            r->draw_line(0.0, y, 0.0, std::max(y - dash, y1), 1.4f, c);

        Rendering::LayerScope txt(r, Rendering::Layer::Text);
        int sx, sy;
        r->world_to_screen(0.0, y1, &sx, &sy);
        const int w = r->measure_text("ref", 12);
        r->draw_text("ref", sx - w - 7, sy - 5, 12, c);
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
            Vector2d force_n = {-std::sin(theta), std::cos(theta)};
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
    double m_t = 0.0;
    double m_kick_at = 0.0, m_kick_period = 0.0;
    double m_kick_dur = 0.30, m_kick_amp = 0.0, m_kick_until = -1.0;
    double m_force_shown = 0.0, m_force_vel = 0.0, m_force_peak = 0.0;

    TunerTarget m_tuner_target = TunerTarget::Angle;
    TunerParam m_tuner_param = TunerParam::Kp;

    PlotWidget m_plot_angle, m_plot_cart, m_plot_force, m_plot_energy;
};

} // namespace manifold::Demo
