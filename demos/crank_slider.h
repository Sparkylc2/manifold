#pragma once

#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class CrankSliderDemo : public DemoBase {
  public:
    static constexpr double MotorRadius = 1.2;
    static constexpr double MotorMass = 20.0;
    static constexpr double ArmLength = 3.0;
    static constexpr double ArmWidth = 0.08;
    static constexpr double ArmMass = 1.0;
    static constexpr double SliderRadius = 0.5;
    static constexpr double SliderMass = 3.0;
    static constexpr double SpringK = 40.0;
    static constexpr double SpringDamp = 1.0;
    static constexpr double SpringRestLen = 2.2;
    static constexpr double MotorSpeed = 2.0; // rad/s
    static constexpr int SimSteps = 50;

    const char *name() const override { return "Crank-Slider"; }
    double default_cam_x() const override { return 2.0; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 55.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        // motor disk - large, heavy
        m_motor.reset();
        m_motor.m = MotorMass;
        m_motor.I = 0.5 * MotorMass * MotorRadius * MotorRadius;
        m_motor.p = Vector2d(0, 0);
        m_motor.theta = 0;
        m_system.add_body(&m_motor);

        // crank arm
        double arm_angle = 0;
        m_arm.reset();
        m_arm.m = ArmMass;
        m_arm.I = ArmMass * ArmLength * ArmLength / 12.0;
        m_arm.p = Vector2d(MotorRadius * 0.9 + ArmLength / 2.0, 0);
        m_arm.theta = 0;
        m_system.add_body(&m_arm);

        // slider
        m_slider.reset();
        m_slider.m = SliderMass;
        m_slider.I = 0.5 * SliderMass * SliderRadius * SliderRadius;
        m_slider.p =
            Vector2d(MotorRadius + ArmLength + SpringRestLen + SliderRadius, 0);
        m_slider.theta = 0;
        m_system.add_body(&m_slider);

        // -- constraints --

        // pin motor at origin
        m_motor_pin.set_body(&m_motor);
        m_motor_pin.set_world_position(Vector2d(0, 0));
        m_motor_pin.set_local_position(Vector2d(0, 0));
        m_motor_pin.set_ks(100.0);
        m_motor_pin.set_kd(10.0);
        m_system.add_constraint(&m_motor_pin);

        m_arm_rail.set_body(&m_arm);
        m_arm_rail.set_line(Vector2d(0, 0), Vector2d(1, 0));
        m_arm_rail.set_local_pos(Vector2d(ArmLength / 2.0, 0)); // arm tip
        m_arm_rail.set_ks(100.0);
        m_arm_rail.set_kd(10.0);
        m_system.add_constraint(&m_arm_rail);

        // link arm to motor edge
        m_crank_link.set_bodies(&m_motor, &m_arm);
        m_crank_link.set_local_pos1(Vector2d(MotorRadius, 0)); // edge of disk
        m_crank_link.set_local_pos2(Vector2d(-ArmLength / 2.0, 0)); // arm start
        m_crank_link.set_ks(100.0);
        m_crank_link.set_kd(10.0);
        m_system.add_constraint(&m_crank_link);

        // slider on horizontal rail
        m_rail.set_body(&m_slider);
        m_rail.set_line(Vector2d(0, 0), Vector2d(1, 0));
        m_rail.set_local_pos(Vector2d(0, 0));
        m_rail.set_ks(100.0);
        m_rail.set_kd(10.0);
        m_system.add_constraint(&m_rail);

        // prevent slider rotation
        m_slider_rot.set_body(&m_slider);
        m_slider_rot.set_angle(0);
        m_slider_rot.set_ks(100.0);
        m_slider_rot.set_kd(10.0);
        m_system.add_constraint(&m_slider_rot);

        // spring from arm tip to slider
        m_spring.set_bodies(&m_arm, &m_slider);
        m_spring.set_local_pos1(Vector2d(ArmLength / 2.0, 0)); // arm end
        m_spring.set_local_pos2(Vector2d(0, 0));
        m_spring.set_rest_length(SpringRestLen);
        m_spring.set_ks(SpringK);
        m_spring.set_kd(SpringDamp);
        m_system.add_force_generator(&m_spring);

        m_plot_slider_x.configure("Slider X (m)",
                                  Rendering::palette::accent2());
        m_plot_spring_f.configure("Spring Force (N)",
                                  Rendering::palette::accent1());
        m_plot_motor_angle.configure("Motor Angle (rad)",
                                     Rendering::palette::accent3());
        m_plot_slider_x.clear();
        m_plot_spring_f.clear();
        m_plot_motor_angle.clear();
    }

    void process(double dt) override {
        // drive motor at constant angular velocity
        m_motor.v_theta = MotorSpeed * m_motor_speed_mult;

        m_system.process(dt, SimSteps);

        m_plot_slider_x.push(m_slider.p.x());
        m_plot_spring_f.push(m_spring.energy() > 0
                                 ? std::sqrt(2.0 * SpringK * m_spring.energy())
                                 : 0);
        m_plot_motor_angle.push(std::fmod(m_motor.theta, 2.0 * M_PI));
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        auto fg = Rendering::palette::foreground();
        auto shadow = Rendering::palette::shadow();
        auto dim = Rendering::palette::text_dim();
        auto accent1 = Rendering::palette::accent1();
        auto accent2 = Rendering::palette::accent2();
        auto accent3 = Rendering::palette::accent3();

        // motor rotation trace (red circle)
        Rendering::draw_arc(r, 0, 0, MotorRadius, 0, 2.0 * M_PI, 2.0f, accent1,
                            64);

        // rail
        r->draw_line(-3, 0, 12, 0, 1.5f, dim);

        // motor disk
        r->draw_disk(m_motor.p.x(), m_motor.p.y(), m_motor.theta, MotorRadius,
                     fg, shadow);

        // crank pin on motor edge
        Vector2d crank_point;
        m_motor.local_to_world(Vector2d(MotorRadius, 0), &crank_point);
        r->draw_circle(crank_point.x(), crank_point.y(), 0.06, accent1);

        // crank arm
        r->draw_bar(m_arm.p.x(), m_arm.p.y(), m_arm.theta, ArmLength, ArmWidth,
                    fg, shadow);

        // arm pivot dots
        Vector2d arm_start, arm_end;
        m_arm.local_to_world(Vector2d(-ArmLength / 2.0, 0), &arm_start);
        m_arm.local_to_world(Vector2d(ArmLength / 2.0, 0), &arm_end);
        r->draw_circle(arm_start.x(), arm_start.y(), 0.04,
                       Rendering::palette::background());
        r->draw_circle(arm_end.x(), arm_end.y(), 0.04,
                       Rendering::palette::background());

        // spring coil between arm end and slider
        draw_spring_coil(r, arm_end.x(), arm_end.y(), m_slider.p.x(),
                         m_slider.p.y(), 8, 0.15);

        // slider disk
        r->draw_disk(m_slider.p.x(), m_slider.p.y(), m_slider.theta,
                     SliderRadius, fg, shadow);

        // motor center pin
        r->draw_circle(0, 0, 0.05, Rendering::palette::background());

        // ---- annotations ----

        // motor angle arc
        Rendering::draw_angle_marker(
            r, 0, 0, 0, m_motor.theta, MotorRadius * 0.5, 1.5f, accent1, dim,
            {.show_label = true, .ref_line_len = MotorRadius * 0.7});

        // slider displacement from rest position
        double rest_x = MotorRadius + ArmLength + SpringRestLen;
        double slider_x = m_slider.p.x();
        if (std::abs(slider_x - rest_x) > 0.1) {
            Rendering::draw_displacement(r, rest_x, 0, slider_x, 0, "dx = %.2f",
                                         slider_x - rest_x, 1.5f, accent3,
                                         -SliderRadius - 0.3);
        }

        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_slider_x, &m_plot_spring_f,
                                           &m_plot_motor_angle};
        render_plots(r, plots, 280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::W))
            m_motor_speed_mult += 0.5;
        if (r->is_key_pressed(Rendering::keys::S))
            m_motor_speed_mult = std::max(0.0, m_motor_speed_mult - 0.5);
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
            r->draw_line(prev_x, prev_y, cx, cy, 2.0f,
                         Rendering::palette::foreground());
            prev_x = cx;
            prev_y = cy;
        }
    }

    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("CRANK-SLIDER", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Motor:    %.2f rad/s",
                 m_motor.v_theta);
        hud.line(Rendering::palette::text(), "Angle:    %.1f deg",
                 std::fmod(m_motor.theta * 180.0 / M_PI, 360.0));
        hud.line(Rendering::palette::text(), "Slider X: %.3f m",
                 m_slider.p.x());
        hud.line(Rendering::palette::text(), "Spring E: %.3f J",
                 m_spring.energy());
        hud.line(Rendering::palette::text(), "Speed:    %.1fx",
                 m_motor_speed_mult);
        hud.separator();
        hud.small_text("[W/S] Motor speed  [R] Reset  [H] Home",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_motor, m_arm, m_slider;

    Solver::FixedPositionConstraint m_motor_pin;
    Solver::LinkConstraint m_crank_link;
    Solver::LineConstraint m_rail;
    Solver::LineConstraint m_arm_rail;
    Solver::FixedRotationConstraint m_slider_rot;
    Solver::Spring m_spring;

    PlotWidget m_plot_slider_x, m_plot_spring_f, m_plot_motor_angle;
    double m_motor_speed_mult = 1.0;
};

} // namespace manifold::Demo
