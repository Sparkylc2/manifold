#pragma once

#include "manifold/renderer/theme.h"
#include <deque>
#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/constant_speed_motor.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gauss_seidel_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class EngineDemo : public DemoBase {
  public:
    // V configuration
    static constexpr double V_angle = M_PI / 2.0;
    static constexpr double V_half = V_angle / 2.0;

    // geometry
    static constexpr double CrankR = 0.7;
    static constexpr double FlyR = 0.9;
    static constexpr double FlyMass = 12.0;
    static constexpr double RodLen = 2.2;
    static constexpr double RodW = 0.06;
    static constexpr double RodMass = 0.8;
    static constexpr double PistonW = 0.55;
    static constexpr double PistonH = 0.3;
    static constexpr double PistonMass = 2.5;
    static constexpr double CylGap = 0.06;

    // forces
    static constexpr double SpringK = 20.0;
    static constexpr double SpringDamp = 1.5;
    static constexpr double SpringRest = 1.2;
    static constexpr double Grav = 9.81;
    static constexpr double MotorSpd = 4.0;

    static constexpr int SimSteps = 60;
    static constexpr int PhaseTrailLen = 300;

    const char *name() const override { return "V-Twin Engine"; }
    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 1.8; }
    double default_cam_zoom() const override { return 50.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        // cylinder axis directions
        m_dir_L = Vector2d(-std::sin(V_half), std::cos(V_half));
        m_dir_R = Vector2d(std::sin(V_half), std::cos(V_half));

        // ---- bodies ----

        // crankshaft / flywheel
        m_crank.reset();
        m_crank.m = FlyMass;
        m_crank.I = 0.5 * FlyMass * FlyR * FlyR;
        m_crank.p = Vector2d(0, 0);
        m_crank.theta = M_PI / 2.0; // start with pin pointing up (symmetric)
        m_system.add_body(&m_crank);

        // crank pin position at startup
        Vector2d pin0(CrankR * std::cos(m_crank.theta),
                      CrankR * std::sin(m_crank.theta));

        // compute initial piston positions along each cylinder axis
        auto init_piston = [&](const Vector2d &dir) -> double {
            double b = -2.0 * dir.dot(pin0);
            double c = pin0.squaredNorm() - RodLen * RodLen;
            return (-b + std::sqrt(b * b - 4.0 * c)) / 2.0;
        };

        double t_L = init_piston(m_dir_L);
        double t_R = init_piston(m_dir_R);

        // left connecting rod
        Vector2d piston_L = t_L * m_dir_L;
        Vector2d rod_L_center = (pin0 + piston_L) * 0.5;
        double rod_L_angle =
            std::atan2(piston_L.y() - pin0.y(), piston_L.x() - pin0.x());

        m_rod_L.reset();
        m_rod_L.m = RodMass;
        m_rod_L.I = RodMass * RodLen * RodLen / 12.0;
        m_rod_L.p = rod_L_center;
        m_rod_L.theta = rod_L_angle;
        m_system.add_body(&m_rod_L);

        // right connecting rod
        Vector2d piston_R = t_R * m_dir_R;
        Vector2d rod_R_center = (pin0 + piston_R) * 0.5;
        double rod_R_angle =
            std::atan2(piston_R.y() - pin0.y(), piston_R.x() - pin0.x());

        m_rod_R.reset();
        m_rod_R.m = RodMass;
        m_rod_R.I = RodMass * RodLen * RodLen / 12.0;
        m_rod_R.p = rod_R_center;
        m_rod_R.theta = rod_R_angle;
        m_system.add_body(&m_rod_R);

        // left piston
        double piston_angle_L = M_PI / 2 - std::atan2(m_dir_L.y(), m_dir_L.x());
        m_piston_L.reset();
        m_piston_L.m = PistonMass;
        m_piston_L.I =
            PistonMass * (PistonW * PistonW + PistonH * PistonH) / 12.0;
        m_piston_L.p = piston_L;
        m_piston_L.theta = piston_angle_L;
        m_system.add_body(&m_piston_L);

        // right piston
        double piston_angle_R = M_PI / 2 - std::atan2(m_dir_R.y(), m_dir_R.x());

        m_piston_R.reset();
        m_piston_R.m = PistonMass;
        m_piston_R.I =
            PistonMass * (PistonW * PistonW + PistonH * PistonH) / 12.0;
        m_piston_R.p = piston_R;
        m_piston_R.theta = piston_angle_R;
        m_system.add_body(&m_piston_R);

        // spring anchors (not in system)
        double anchor_dist = CrankR + RodLen + SpringRest + 0.5;
        m_sp_anchor_L.reset();
        m_sp_anchor_L.p = anchor_dist * m_dir_L;
        m_sp_anchor_L.m = 1.0;
        m_sp_anchor_L.I = 1.0;

        m_sp_anchor_R.reset();
        m_sp_anchor_R.p = anchor_dist * m_dir_R;
        m_sp_anchor_R.m = 1.0;
        m_sp_anchor_R.I = 1.0;

        // ---- constraints ----

        // pin crankshaft
        m_crank_pin.set_body(&m_crank);
        m_crank_pin.set_world_position(Vector2d(0, 0));
        m_crank_pin.set_local_position(Vector2d(0, 0));
        m_crank_pin.set_ks(1000.0);
        m_crank_pin.set_kd(100.0);
        m_system.add_constraint(&m_crank_pin);

        // big ends — both rods connect to same crank pin
        m_big_end_L.set_bodies(&m_crank, &m_rod_L);
        m_big_end_L.set_local_pos1(Vector2d(CrankR, 0));
        m_big_end_L.set_local_pos2(Vector2d(-RodLen / 2.0, 0));
        m_big_end_L.set_ks(100.0);
        m_big_end_L.set_kd(10.0);
        m_system.add_constraint(&m_big_end_L);

        m_big_end_R.set_bodies(&m_crank, &m_rod_R);
        m_big_end_R.set_local_pos1(Vector2d(CrankR, 0));
        m_big_end_R.set_local_pos2(Vector2d(-RodLen / 2.0, 0));
        m_big_end_R.set_ks(100.0);
        m_big_end_R.set_kd(10.0);
        m_system.add_constraint(&m_big_end_R);

        // small ends — rod to piston
        m_small_end_L.set_bodies(&m_rod_L, &m_piston_L);
        m_small_end_L.set_local_pos1(Vector2d(RodLen / 2.0, 0));
        m_small_end_L.set_local_pos2(Vector2d(0, 0));
        m_small_end_L.set_ks(100.0);
        m_small_end_L.set_kd(10.0);
        m_system.add_constraint(&m_small_end_L);

        m_small_end_R.set_bodies(&m_rod_R, &m_piston_R);
        m_small_end_R.set_local_pos1(Vector2d(RodLen / 2.0, 0));
        m_small_end_R.set_local_pos2(Vector2d(0, 0));
        m_small_end_R.set_ks(100.0);
        m_small_end_R.set_kd(10.0);
        m_system.add_constraint(&m_small_end_R);

        // cylinder line constraints
        m_cyl_L.set_body(&m_piston_L);
        m_cyl_L.set_line(Vector2d(0, 0), m_dir_L);
        m_cyl_L.set_local_pos(Vector2d(0, 0));
        m_cyl_L.set_ks(100.0);
        m_cyl_L.set_kd(10.0);
        m_system.add_constraint(&m_cyl_L);

        m_cyl_R.set_body(&m_piston_R);
        m_cyl_R.set_line(Vector2d(0, 0), m_dir_R);
        m_cyl_R.set_local_pos(Vector2d(0, 0));
        m_cyl_R.set_ks(100.0);
        m_cyl_R.set_kd(10.0);
        m_system.add_constraint(&m_cyl_R);

        // prevent piston rotation
        m_rot_L.set_body(&m_piston_L);
        m_rot_L.set_angle(piston_angle_L);
        m_rot_L.set_ks(100.0);
        m_rot_L.set_kd(10.0);
        m_system.add_constraint(&m_rot_L);

        m_rot_R.set_body(&m_piston_R);
        m_rot_R.set_angle(piston_angle_R);
        m_rot_R.set_ks(100.0);
        m_rot_R.set_kd(10.0);
        m_system.add_constraint(&m_rot_R);

        // ---- forces ----

        m_motor.set_bodies(nullptr, &m_crank);
        m_motor.set_speed(MotorSpd);
        m_motor.set_max_torque(800.0);
        m_motor.set_ks(80.0);
        m_motor.set_kd(5.0);
        m_system.add_force_generator(&m_motor);

        m_gravity.set_gravity(Grav);
        m_system.add_force_generator(&m_gravity);

        m_spring_L.set_bodies(&m_piston_L, &m_sp_anchor_L);
        m_spring_L.set_local_pos1(Vector2d(0, 0));
        m_spring_L.set_local_pos2(Vector2d(0, 0));
        m_spring_L.set_rest_length(SpringRest);
        m_spring_L.set_ks(SpringK);
        m_spring_L.set_kd(SpringDamp);
        m_system.add_force_generator(&m_spring_L);

        m_spring_R.set_bodies(&m_piston_R, &m_sp_anchor_R);
        m_spring_R.set_local_pos1(Vector2d(0, 0));
        m_spring_R.set_local_pos2(Vector2d(0, 0));
        m_spring_R.set_rest_length(SpringRest);
        m_spring_R.set_ks(SpringK);
        m_spring_R.set_kd(SpringDamp);
        m_system.add_force_generator(&m_spring_R);

        // plots
        m_plot_L.configure("Left Piston", Rendering::palette::accent2());
        m_plot_R.configure("Right Piston", Rendering::palette::accent3());
        m_plot_rpm.configure("RPM", Rendering::palette::accent1());
        m_plot_L.clear();
        m_plot_R.clear();
        m_plot_rpm.clear();
        m_phase_trail.clear();

        m_speed_mult = 1.0;
    }

    void process(double dt) override {
        m_motor.set_speed(MotorSpd * m_speed_mult);
        m_system.process(dt, SimSteps);

        // piston displacement along cylinder axis (distance from origin)
        double disp_L = m_piston_L.p.dot(m_dir_L);
        double disp_R = m_piston_R.p.dot(m_dir_R);

        m_plot_L.push(disp_L);
        m_plot_R.push(disp_R);
        m_plot_rpm.push(m_crank.v_theta * 60.0 / (2.0 * M_PI));

        // phase trail
        m_phase_trail.push_back({disp_L, disp_R});
        while ((int)m_phase_trail.size() > PhaseTrailLen)
            m_phase_trail.pop_front();
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        render_cell(r);

        draw_phase_diagram(r);

        render_hud(r);

        std::vector<PlotWidget *> plots = {&m_plot_L, &m_plot_R, &m_plot_rpm};
        render_plots(r, plots, 280, 65);
    }

    void render_cell(Rendering::Renderer *r) override {

        auto fg = Rendering::palette::foreground();
        auto dim = Rendering::palette::text_dim();
        auto shadow = Rendering::palette::shadow();
        auto a1 = Rendering::palette::accent1();
        auto a2 = Rendering::palette::accent2();
        auto a3 = Rendering::palette::accent3();

        double tdc_dist = CrankR + RodLen;
        double bdc_dist = RodLen - CrankR;

        // ---- cylinder walls ----
        // perpendicular to each cylinder direction
        Vector2d perp_L(-m_dir_L.y(), m_dir_L.x());
        Vector2d perp_R(-m_dir_R.y(), m_dir_R.x());

        double cyl_half = PistonW / 2.0 + CylGap;
        double cyl_bot = -0.3;
        double cyl_top = tdc_dist + 0.6;

        for (int side : {-1, 1}) {
            // left cylinder walls
            Vector2d lw0 = m_dir_L * cyl_bot + perp_L * (side * cyl_half);
            Vector2d lw1 = m_dir_L * cyl_top + perp_L * (side * cyl_half);
            r->draw_line(lw0.x(), lw0.y(), lw1.x(), lw1.y(), 2.0f, dim);

            // right cylinder walls
            Vector2d rw0 = m_dir_R * cyl_bot + perp_R * (side * cyl_half);
            Vector2d rw1 = m_dir_R * cyl_top + perp_R * (side * cyl_half);
            r->draw_line(rw0.x(), rw0.y(), rw1.x(), rw1.y(), 2.0f, dim);
        }

        // cylinder heads
        Vector2d lh0 = m_dir_L * cyl_top + perp_L * cyl_half;
        Vector2d lh1 = m_dir_L * cyl_top - perp_L * cyl_half;
        r->draw_line(lh0.x(), lh0.y(), lh1.x(), lh1.y(), 2.0f, dim);

        Vector2d rh0 = m_dir_R * cyl_top + perp_R * cyl_half;
        Vector2d rh1 = m_dir_R * cyl_top - perp_R * cyl_half;
        r->draw_line(rh0.x(), rh0.y(), rh1.x(), rh1.y(), 2.0f, dim);

        // ---- TDC / BDC reference lines (dashed) ----
        for (auto *dir : {&m_dir_L, &m_dir_R}) {
            auto *perp = (dir == &m_dir_L) ? &perp_L : &perp_R;
            int side = (dir == &m_dir_L) ? 1 : 0;
            double gap = (dir == &m_dir_L) ? 2 : 0.1;

            Vector2d tdc0 = *dir * tdc_dist + *perp * (cyl_half + 0.1);
            Vector2d tdc1 = *dir * tdc_dist - *perp * (cyl_half + 0.1);
            Rendering::label_with_line(r, tdc0.x(), tdc0.y(), tdc1.x(),
                                       tdc1.y(), "TDC", 11, dim, dim,
                                       {
                                           .gap = 0.05,
                                           .text_end = side,
                                           .line_width = 1.2f,
                                           .space = 0.05,
                                       });

            Vector2d bdc0 = *dir * bdc_dist + *perp * (cyl_half + 0.1);
            Vector2d bdc1 = *dir * bdc_dist - *perp * (cyl_half + 0.1);
            Rendering::label_with_line(
                r, bdc0.x(), bdc0.y(), bdc1.x(), bdc1.y(), "BDC", 11, dim, dim,
                {.text_end = side, .line_width = 1.2f, .space = 0.1});
        }

        // ---- piston offset annotation ----
        for (auto *dir : {&m_dir_L, &m_dir_R}) {
            auto *perp = (dir == &m_dir_L) ? &perp_L : &perp_R;
            const Solver::RigidBody &body =
                (dir == &m_dir_L) ? m_piston_L : m_piston_R;

            // measure from BDC along the cylinder axis to the piston
            Vector2d bdc_pt = *dir * bdc_dist;
            double x = (body.p - bdc_pt).dot(*dir); // signed travel from BDC

            // axis angle of this cylinder (direction of piston travel)
            double axis_angle = std::atan2(dir->y(), dir->x());

            // offset sign: push the annotation to the outboard side of each
            // bank
            double off = (dir == &m_dir_L) ? -0.75 : 0.75;

            Rendering::draw_displacement(r, bdc_pt.x(), bdc_pt.y(), body.p.x(),
                                         body.p.y(), "x = %.3f m", x, 2.0f,
                                         Rendering::palette::accent1(),
                                         axis_angle, {.offset = off});
        }

        // ---- crankshaft / flywheel ----
        Rendering::draw_body_disk(r, 0, 0, FlyR, m_crank.theta);

        // counterweight (thick arc opposite the crank pin)
        double cw_start = m_crank.theta + M_PI - 0.6;
        double cw_end = m_crank.theta + M_PI + 0.6;
        int cw_segs = 12;
        for (int i = 0; i < cw_segs; ++i) {
            double a0 = cw_start + (cw_end - cw_start) * i / cw_segs;
            double a1_cw = cw_start + (cw_end - cw_start) * (i + 1) / cw_segs;
            for (double cr = FlyR * 0.5; cr <= FlyR * 0.85; cr += 0.04) {
                r->draw_line(std::cos(a0) * cr, std::sin(a0) * cr,
                             std::cos(a1_cw) * cr, std::sin(a1_cw) * cr, 1.0f,
                             fg);
            }
        }

        // crank arm
        Vector2d pin;
        m_crank.local_to_world(Vector2d(CrankR, 0), &pin);
        r->draw_line(0, 0, pin.x(), pin.y(), 3.0f, fg);

        // crank rotation trace
        Rendering::draw_arc(r, 0, 0, CrankR, 0, 2.0 * M_PI, 1.0f, dim, 48);

        // ---- connecting rods ----
        Rendering::draw_body_bar(r, m_rod_L.p.x(), m_rod_L.p.y(), RodLen, RodW,
                                 m_rod_L.theta);
        Rendering::draw_body_bar(r, m_rod_R.p.x(), m_rod_R.p.y(), RodLen, RodW,
                                 m_rod_R.theta);

        // ---- pistons ----
        Rendering::draw_body_block(r, m_piston_L.p.x(), m_piston_L.p.y(),
                                   PistonH, PistonW, m_piston_L.theta);
        Rendering::draw_body_block(r, m_piston_R.p.x(), m_piston_R.p.y(),
                                   PistonH, PistonW, m_piston_R.theta);

        // ---- springs ----
        Vector2d sp_L0, sp_L1, sp_R0, sp_R1;
        m_spring_L.get_ends(&sp_L0, &sp_L1);
        m_spring_R.get_ends(&sp_R0, &sp_R1);
        Rendering::draw_spring(r, sp_L0, sp_L1, {.coils = 6, .amp = 0.08});

        Rendering::draw_spring(r, sp_R0, sp_R1, {.coils = 6, .amp = 0.08});

        // ---- joint markers ----
        Rendering::draw_pivot(r, 0, 0);
        // Rendering::draw_ground_anchor(r, 0, 0, 0.3);
        r->draw_circle(pin.x(), pin.y(), 0.05, fg); // shared crank pin

        // wrist pins
        Vector2d wp_L, wp_R;
        m_rod_L.local_to_world(Vector2d(RodLen / 2.0, 0), &wp_L);
        m_rod_R.local_to_world(Vector2d(RodLen / 2.0, 0), &wp_R);
        r->draw_circle(wp_L.x(), wp_L.y(), 0.04, fg);
        r->draw_circle(wp_R.x(), wp_R.y(), 0.04, fg);

        // ---- annotations ----

        // crank angle
        Rendering::draw_angle_marker(
            r, 0, 0, M_PI / 2.0, m_crank.theta, CrankR * 0.55, 1.5f, a1, dim,
            {.show_label = false, .ref_line_len = CrankR * 0.8});

        // ---- phase diagram ----
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::W))
            m_speed_mult += 0.5;
        if (r->is_key_pressed(Rendering::keys::S))
            m_speed_mult = std::max(0.0, m_speed_mult - 0.5);
    }

  private:
    void draw_phase_diagram(Rendering::Renderer *r) {
        if (m_phase_trail.size() < 2)
            return;

        auto &t = Rendering::active_theme();

        // box position (top-right, or top-left of portrait strip)
        int sw = r->screen_width();
        int box_size = 120;
        int margin = 12;
        int bx = portrait_mode() ? portrait_strip_right(r) - box_size - margin
                                 : sw - box_size - margin;
        int by = margin;

        // background
        r->draw_screen_rect(bx - 2, by - 2, box_size + 4, box_size + 4,
                            t.panel_bg);

        // axis labels
        r->draw_text("L", bx + 2, by + box_size + 2, 10, t.text_dim);
        r->draw_text("R", bx - 12, by + 2, 10, t.text_dim);

        // normalize to BDC..TDC range
        double lo = RodLen - CrankR;
        double hi = RodLen + CrankR;

        auto map_x = [&](double v) {
            return bx + (int)((v - lo) / (hi - lo) * box_size);
        };
        auto map_y = [&](double v) {
            return by + box_size - (int)((v - lo) / (hi - lo) * box_size);
        };

        // trail
        for (int i = 1; i < (int)m_phase_trail.size(); ++i) {
            double alpha = (double)i / m_phase_trail.size();
            auto c = Rendering::Color::rgba(
                (unsigned char)(t.accent2.r * alpha +
                                t.text_dim.r * (1 - alpha)),
                (unsigned char)(t.accent2.g * alpha +
                                t.text_dim.g * (1 - alpha)),
                (unsigned char)(t.accent2.b * alpha +
                                t.text_dim.b * (1 - alpha)),
                (unsigned char)(200 * alpha));

            int x0 = map_x(m_phase_trail[i - 1].first);
            int y0 = map_y(m_phase_trail[i - 1].second);
            int x1 = map_x(m_phase_trail[i].first);
            int y1 = map_y(m_phase_trail[i].second);

            r->draw_screen_line(x0, y0, x1, y1, 1.5f, c);
        }

        // current position dot
        auto &last = m_phase_trail.back();
        int cx = map_x(last.first);
        int cy = map_y(last.second);
        r->draw_screen_rect(cx - 2, cy - 2, 5, 5, t.accent1);
    }

    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("V-TWIN ENGINE", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "RPM:     %.0f",
                 m_crank.v_theta * 60.0 / (2.0 * M_PI));
        hud.line(Rendering::palette::text(), "V-angle: %.0f deg",
                 V_angle * 180.0 / M_PI);
        hud.line(Rendering::palette::accent2(), "Left:    %.2f",
                 m_piston_L.p.dot(m_dir_L));
        hud.line(Rendering::palette::accent3(), "Right:   %.2f",
                 m_piston_R.p.dot(m_dir_R));
        hud.line(Rendering::palette::text(), "Speed:   %.1fx", m_speed_mult);
        hud.separator();
        hud.small_text("[W/S] Speed  [R] Reset  [P] Portrait",
                       Rendering::palette::text_dim());
    }

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussSeidelSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_crank;
    Solver::RigidBody m_rod_L, m_rod_R;
    Solver::RigidBody m_piston_L, m_piston_R;
    Solver::RigidBody m_sp_anchor_L, m_sp_anchor_R;

    Solver::FixedPositionConstraint m_crank_pin;
    Solver::LinkConstraint m_big_end_L, m_big_end_R;
    Solver::LinkConstraint m_small_end_L, m_small_end_R;
    Solver::LineConstraint m_cyl_L, m_cyl_R;
    Solver::FixedRotationConstraint m_rot_L, m_rot_R;

    Solver::ConstantSpeedMotor m_motor;
    Solver::UniformGravityForceGenerator m_gravity;
    Solver::Spring m_spring_L, m_spring_R;

    Vector2d m_dir_L, m_dir_R;
    std::deque<std::pair<double, double>> m_phase_trail;

    PlotWidget m_plot_L, m_plot_R, m_plot_rpm;
    double m_speed_mult = 1.0;
};

} // namespace manifold::Demo
