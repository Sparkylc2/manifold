#pragma once

#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/mouse_spring.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <cmath>

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
    static constexpr int SimSteps = 60;
    static constexpr double GrabRadius = 0.5;

    // initial angles from vertical (positive = right of vertical)
    static constexpr double InitAngle1 = 0.6;
    static constexpr double InitAngle2 = 0.4;

    const char *name() const override { return "Double Pendulum"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return -2.0; }
    double default_cam_zoom() const override { return 50.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        // anchor body (massive, pinned)
        m_anchor.reset();
        m_anchor.m = 1e6;
        m_anchor.I = 1e6;
        m_anchor.p = Vector2d(0, 0);
        m_system.add_body(&m_anchor);

        // bar 1: theta measured from horizontal, hanging = -pi/2
        double theta1 = -M_PI / 2.0 + InitAngle1;
        m_bar1.reset();
        m_bar1.m = M1;
        m_bar1.I = M1 * L1 * L1 / 12.0;
        m_bar1.theta = theta1;
        m_bar1.p =
            Vector2d(std::cos(theta1) * L1 / 2.0, std::sin(theta1) * L1 / 2.0);
        m_system.add_body(&m_bar1);

        // bar 2
        Vector2d joint1;
        m_bar1.local_to_world(Vector2d(-L1 / 2.0, 0), &joint1);
        // joint1 is the lower end of bar1

        double theta2 = -M_PI / 2.0 + InitAngle2;
        m_bar2.reset();
        m_bar2.m = M2;
        m_bar2.I = M2 * L2 * L2 / 12.0;
        m_bar2.theta = theta2;
        m_bar2.p = Vector2d(joint1.x() + std::cos(theta2) * L2 / 2.0,
                            joint1.y() + std::sin(theta2) * L2 / 2.0);
        m_system.add_body(&m_bar2);

        // pin anchor at origin
        m_pin.set_body(&m_anchor);
        m_pin.set_world_position(Vector2d(0, 0));
        m_pin.set_local_position(Vector2d(0, 0));
        m_pin.set_ks(100.0);
        m_pin.set_kd(10.0);
        m_system.add_constraint(&m_pin);

        // link anchor to top of bar1
        m_link1.set_bodies(&m_anchor, &m_bar1);
        m_link1.set_local_pos1(Vector2d(0, 0));
        m_link1.set_local_pos2(Vector2d(L1 / 2.0, 0)); // top end of bar1
        m_link1.set_ks(100.0);
        m_link1.set_kd(10.0);
        m_system.add_constraint(&m_link1);

        // link bottom of bar1 to top of bar2
        m_link2.set_bodies(&m_bar1, &m_bar2);
        m_link2.set_local_pos1(Vector2d(-L1 / 2.0, 0)); // bottom end of bar1
        m_link2.set_local_pos2(Vector2d(L2 / 2.0, 0));  // top end of bar2
        m_link2.set_ks(100.0);
        m_link2.set_kd(10.0);
        m_system.add_constraint(&m_link2);

        m_gravity.set_gravity(Gravity);
        m_system.add_force_generator(&m_gravity);
        m_system.add_force_generator(&m_mouse_spring);

        m_plot_theta1.configure("θ₁ (deg)", Rendering::palette::accent2(), 600,
                                [](double v) { return v * 180.0 / M_PI; });
        m_plot_theta2.configure("θ₂ (deg)", Rendering::palette::accent3(), 600,
                                [](double v) { return v * 180.0 / M_PI; });
        m_plot_energy.configure("Total Energy (J)",
                                Rendering::palette::accent1());
        m_plot_theta1.clear();
        m_plot_theta2.clear();
        m_plot_energy.clear();

        m_trail.clear();
        m_grabbed_body = nullptr;
        m_mouse_spring.set_active(false);
        m_mouse_spring.set_ks(100.0);
    }

    void process(double dt) override {
        m_system.process(dt, SimSteps);

        // angle from vertical: theta_from_vert = theta - (-pi/2) = theta + pi/2
        double t1 = m_bar1.theta + M_PI / 2.0;
        double t2 = m_bar2.theta - m_bar1.theta; // relative angle

        m_plot_theta1.push(t1);
        m_plot_theta2.push(t2);

        double ke = m_bar1.energy() + m_bar2.energy();
        double pe = M1 * Gravity * m_bar1.p.y() + M2 * Gravity * m_bar2.p.y();
        m_plot_energy.push(ke + pe);

        // trail of bar2 tip
        Vector2d tip;
        m_bar2.local_to_world(Vector2d(-L2 / 2.0, 0), &tip);
        m_trail.push_back(tip);
        if ((int)m_trail.size() > 800)
            m_trail.erase(m_trail.begin());
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        auto fg = Rendering::palette::foreground();
        auto shadow = Rendering::palette::shadow();
        auto dim = Rendering::palette::text_dim();
        auto a1 = Rendering::palette::accent1();
        auto a2 = Rendering::palette::accent2();
        auto a3 = Rendering::palette::accent3();
        auto ann = Rendering::Color::rgba(100, 160, 200, 160);

        // tip trail
        for (int i = 1; i < (int)m_trail.size(); ++i) {
            double alpha = (double)i / m_trail.size();
            auto tc = Rendering::Color::rgba(
                (unsigned char)(a1.r * alpha), (unsigned char)(a1.g * alpha),
                (unsigned char)(a1.b * alpha), (unsigned char)(200 * alpha));
            r->draw_line(m_trail[i - 1].x(), m_trail[i - 1].y(), m_trail[i].x(),
                         m_trail[i].y(), 1.5f, tc);
        }

        // get key positions
        Vector2d pivot(0, 0);
        Vector2d joint;
        m_bar1.local_to_world(Vector2d(-L1 / 2.0, 0), &joint);
        Vector2d tip;
        m_bar2.local_to_world(Vector2d(-L2 / 2.0, 0), &tip);
        Vector2d bar1_top;
        m_bar1.local_to_world(Vector2d(L1 / 2.0, 0), &bar1_top);
        Vector2d bar2_top;
        m_bar2.local_to_world(Vector2d(L2 / 2.0, 0), &bar2_top);

        // ---- annotations ----

        // θ₁: angle from vertical at pivot
        double ref_down = -M_PI / 2.0;
        Rendering::draw_angle_marker(r, pivot.x(), pivot.y(), ref_down,
                                     m_bar1.theta, L1 * 0.3, 1.5f, a2, dim,
                                     true, L1 * 0.5);

        // θ₂: angle from bar1's extension at joint
        // reference direction = bar1's direction (extension beyond joint)
        double bar1_dir = m_bar1.theta + M_PI; // extension direction
        Rendering::draw_angle_marker(r, joint.x(), joint.y(), bar1_dir,
                                     m_bar2.theta, L2 * 0.3, 1.5f, a3, dim,
                                     true, L2 * 0.4);

        // x-displacement of bar1 tip along perpendicular to bar
        // show how far the lower end of bar1 has swung from directly below
        // pivot
        double equil_x1 = 0;
        double equil_y1 = -L1;
        if ((joint - Vector2d(equil_x1, equil_y1)).norm() > 0.1) {
            Rendering::draw_dashed_line(r, pivot.x(), pivot.y(), equil_x1,
                                        equil_y1, 1.0f, dim, 0.08, 0.06);
            // horizontal displacement of joint from equilibrium
            Rendering::draw_displacement(r, equil_x1, joint.y(), joint.x(),
                                         joint.y(), "x₁=%.2f", joint.x(), 1.5f,
                                         a2, 0.3);
        }

        // x-displacement of tip from joint's equilibrium
        if (std::abs(tip.x() - joint.x()) > 0.05) {
            Rendering::draw_displacement(r, joint.x(), tip.y(), tip.x(),
                                         tip.y(), "x₂=%.2f",
                                         tip.x() - joint.x(), 1.5f, a3, 0.3);
        }

        // ---- scene ----

        // anchor point
        r->draw_circle(pivot.x(), pivot.y(), 0.12, a1);
        r->draw_circle(pivot.x(), pivot.y(), 0.06,
                       Rendering::palette::background());

        // bar 1
        r->draw_bar(m_bar1.p.x(), m_bar1.p.y(), m_bar1.theta, L1, BarWidth, a2,
                    shadow);

        // joint
        r->draw_circle(joint.x(), joint.y(), 0.08, fg);
        r->draw_circle(joint.x(), joint.y(), 0.04,
                       Rendering::palette::background());

        // bar 2
        r->draw_bar(m_bar2.p.x(), m_bar2.p.y(), m_bar2.theta, L2, BarWidth, a3,
                    shadow);

        // tip
        r->draw_circle(tip.x(), tip.y(), 0.06, fg);

        // mouse spring visual
        if (m_grabbed_body) {
            double mx, my;
            get_mouse_world(r, &mx, &my);
            draw_spring_coil(r, m_grabbed_body->p.x(), m_grabbed_body->p.y(),
                             mx, my, 8, 0.06);
            r->draw_circle(mx, my, 0.03, a3);
        }

        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_theta1, &m_plot_theta2,
                                           &m_plot_energy};
        render_plots(r, plots, 280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space)) {
            m_bar2.v_theta += 5.0;
        }
        if (r->is_key_pressed(Rendering::keys::C))
            m_trail.clear();

        double mx, my;
        get_mouse_world(r, &mx, &my);

        if (r->is_mouse_button_pressed(Rendering::mouse::Left)) {
            double best = GrabRadius;
            m_grabbed_body = nullptr;

            for (auto *b : {&m_bar1, &m_bar2}) {
                // check distance to bar center and both endpoints
                double d = (b->p - Vector2d(mx, my)).norm();
                if (d < best) {
                    best = d;
                    m_grabbed_body = b;
                }
            }

            if (m_grabbed_body) {
                m_mouse_spring.set_active(true);
                m_mouse_spring.set_body(m_grabbed_body);
                m_mouse_spring.set_target(Vector2d(mx, my));
            }
        }

        if (m_grabbed_body && r->is_mouse_button_down(Rendering::mouse::Left))
            m_mouse_spring.set_target(Vector2d(mx, my));

        if (m_grabbed_body &&
            !r->is_mouse_button_down(Rendering::mouse::Left)) {
            m_grabbed_body = nullptr;
            m_mouse_spring.set_active(false);
        }
    }

  private:
    void draw_spring_coil(Rendering::Renderer *r, double x0, double y0,
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
            double frac = (double)i / segs;
            double across = 0;
            int phase = i % 4;
            if (phase == 1)
                across = amp;
            else if (phase == 3)
                across = -amp;
            double cx = x0 + dx * frac + px * across;
            double cy = y0 + dy * frac + py * across;
            r->draw_line(prev_x, prev_y, cx, cy, 1.5f,
                         Rendering::palette::accent3());
            prev_x = cx;
            prev_y = cy;
        }
    }

    void get_mouse_world(Rendering::Renderer *r, double *wx, double *wy) {
        int mx, my;
        r->get_mouse_pos(&mx, &my);
        r->screen_to_world(mx, my, wx, wy);
    }

    void render_hud(Rendering::Renderer *r) {
        double t1 = m_bar1.theta + M_PI / 2.0;
        double t2 = m_bar2.theta - m_bar1.theta;
        double ke = m_bar1.energy() + m_bar2.energy();
        double pe = M1 * Gravity * m_bar1.p.y() + M2 * Gravity * m_bar2.p.y();

        Vector2d tip;
        m_bar2.local_to_world(Vector2d(-L2 / 2.0, 0), &tip);

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("DOUBLE PENDULUM", Rendering::palette::accent2());
        hud.line(Rendering::palette::accent2(), "θ₁:     %.2f deg",
                 t1 * 180.0 / M_PI);
        hud.line(Rendering::palette::accent3(), "θ₂:     %.2f deg",
                 t2 * 180.0 / M_PI);
        hud.line(Rendering::palette::text(), "Tip:    (%.2f, %.2f)", tip.x(),
                 tip.y());
        hud.line(Rendering::palette::text(), "KE:     %.3f J", ke);
        hud.line(Rendering::palette::text(), "PE:     %.3f J", pe);
        hud.line(Rendering::palette::accent1(), "Total:  %.3f J", ke + pe);

        if (m_grabbed_body) {
            hud.separator();
            hud.line(Rendering::palette::accent3(), "Pulling...");
        }

        hud.separator();
        hud.small_text("[LMB] Pull  [SPACE] Kick  [C] Clear trail",
                       Rendering::palette::text_dim());
        hud.small_text("[R] Reset  [H] Home", Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_anchor, m_bar1, m_bar2;
    Solver::FixedPositionConstraint m_pin;
    Solver::LinkConstraint m_link1, m_link2;
    Solver::UniformGravityForceGenerator m_gravity;
    Solver::MouseSpringForceGenerator m_mouse_spring;

    Solver::RigidBody *m_grabbed_body = nullptr;
    std::vector<Vector2d> m_trail;

    PlotWidget m_plot_theta1, m_plot_theta2, m_plot_energy;
};

} // namespace manifold::Demo
