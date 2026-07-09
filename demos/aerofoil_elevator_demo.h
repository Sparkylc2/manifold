#pragma once

#include <manifold/control/elevator_servo.h>
#include <manifold/coupling/fluid_wrench_force.h>
#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/fluid/mac_fluid_solver.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/aero_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/force_generator.h>
#include <manifold/solver/forces/mouse_spring.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/forces/torsion_spring.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include "manifold/renderer/theme.h"
#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class AerofoilElevatorDemo : public DemoBase {
  public:
    static constexpr int COLS = 250;
    static constexpr int ROWS = 100;
    static constexpr double CELL = 0.055;
    static constexpr int SS = 2;

    static constexpr double INFLOW = 12.0;
    static constexpr double CHORD = 2;
    static constexpr int PANELS = 32;
    static constexpr int RENDER_PANELS = 64;

    // main foil
    static constexpr int FOIL_CODE = 2412;
    static constexpr double MASS = 0.5;
    static constexpr double SPRING_K = 40.0;
    static constexpr double SPRING_KU = 0.1;

    static constexpr double ANCHOR_L = 1.6;
    static constexpr double TORSION_K = 0;
    static constexpr double TORSION_C = 0.0;
    static constexpr double AOA0 = -0.5;
    static constexpr int SUBSTEPS = 8;

    static constexpr int ELEV_CODE = 12; // NACA 0012
    static constexpr double ELEV_CHORD = 1.0;
    static constexpr double ELEV_MASS = 0.12;
    // tail boom: hold the elevator well aft of the foil wake, on a long lever,
    // so a small deflection has real pitch authority over the assembly
    static constexpr double HINGE_GAP = 0.9; // TE -> elevator LE clearance

    static constexpr double SERVO_KP = 25.0;
    static constexpr double SERVO_KD = 1.5;
    static constexpr double SERVO_TAU_MAX = 40.0;
    static constexpr double CMD_MAX = 0.5;    // max deflection (rad)
    static constexpr double CMD_RATE = 1.6;   // command slew (rad/s)
    static constexpr double CMD_RETURN = 2.5; // self-center rate (rad/s)

    static constexpr int BRUSH = 2;
    static constexpr double DENS_RATE = 240;

    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int FADE_PX = 16;

    const char *name() const override { return "Aerofoil + Elevator"; }

    void initialize() override {
        m_stam.clear();
        m_stam.set_channel(INFLOW);
        m_mac.clear();
        m_mac.set_channel(INFLOW);
        m_mac.set_smoke(true);
        m_fluid = m_use_mac ? (Fluid::FluidSolver *)&m_mac
                            : (Fluid::FluidSolver *)&m_stam;

        const Vector2d o = m_fluid->origin();
        m_rest = o + Vector2d(0.30 * COLS * CELL, 0.5 * ROWS * CELL);

        m_foil.reset();
        m_foil.m = MASS;
        m_foil.p = m_rest;
        m_foil.theta = AOA0;
        build_foil();

        // pin foil TE to elevator LE, both starting aligned (zero deflection)
        m_elev.reset();
        m_elev.m = ELEV_MASS;
        m_elev.theta = AOA0;
        build_elevator();
        place_elevator();

        m_anchor_x.reset();
        m_anchor_x.p = m_rest + Vector2d(-ANCHOR_L, 0.0);
        m_anchor_y.reset();
        m_anchor_y.p = m_rest + Vector2d(0.0, -ANCHOR_L);

        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);
        m_system.add_body(&m_foil);
        m_system.add_body(&m_elev);

        for (auto *sp : {&m_spring_x, &m_spring_y}) {
            sp->set_local_pos1(Vector2d::Zero());
            sp->set_local_pos2(Vector2d::Zero());
            sp->set_rest_length(ANCHOR_L);
            sp->set_ks((sp == &m_spring_x ? SPRING_K : SPRING_KU));
            sp->set_kd(0.0);
        }
        m_spring_x.set_bodies(&m_anchor_x, &m_foil);
        m_spring_y.set_bodies(&m_anchor_y, &m_foil);
        m_system.add_force_generator(&m_spring_x);
        m_system.add_force_generator(&m_spring_y);

        m_torsion.set_body(&m_foil);
        m_torsion.set_rest_angle(AOA0 / 2.0);
        m_torsion.set_ks(TORSION_K);
        m_torsion.set_kd(TORSION_C);
        m_system.add_force_generator(&m_torsion);

        // revolute hinge: foil local anchor coincides with elevator local
        // anchor
        m_hinge.set_bodies(&m_foil, &m_elev);
        m_hinge.set_local_pos1(m_foil_anchor);
        m_hinge.set_local_pos2(m_elev_anchor);
        m_hinge.set_ks(120.0);
        m_hinge.set_kd(12.0);
        m_system.add_constraint(&m_hinge);

        // servo torque acts +on elevator, -on foil (reaction into the foil)
        m_servo.set_bodies(&m_elev, &m_foil);
        m_servo.set_gains(SERVO_KP, SERVO_KD, SERVO_TAU_MAX);
        m_system.add_force_generator(&m_servo);

        m_mouse.set_body(&m_foil);
        m_mouse.set_local(Vector2d::Zero());
        m_mouse.set_ks(10.0);
        m_mouse.set_kd(5.0);
        m_mouse.set_active(false);
        m_system.add_force_generator(&m_mouse);

        m_system.add_force_generator(&m_foil_force);
        m_system.add_force_generator(&m_elev_force);
        for (auto *f :
             {(Fluid::FluidSolver *)&m_stam, (Fluid::FluidSolver *)&m_mac}) {
            f->add_boundary(&m_foil_boundary);
            f->add_boundary(&m_elev_boundary);
        }

        m_field.init(COLS, ROWS,
                     {.supersample = SS,
                      .edge_fade_px = FADE_PX,
                      .gamma = 0.29,
                      .colorbar = true},
                     Rendering::speed_ramp());
        m_field.set_scale(0.0, 2.0 * INFLOW, "speed");

        m_cmd = 0.0;
    }

    void process(double dt) override {
        for (int k = 1; k <= 6; ++k) {
            int cj = ROWS * k / 7;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = 3; di <= 9; di++)
                    m_fluid->add_density_source(di, cj + dj, 90.0);
        }

        m_fluid->advance(dt);

        Vector2d Ff, Fe;
        double tf, te;
        m_fluid->wrench_on(m_foil_boundary, m_foil.p, &Ff, &tf);
        m_fluid->wrench_on(m_elev_boundary, m_elev.p, &Fe, &te);
        m_last_F = Ff;
        m_last_tau = tf;
        m_foil_force.set_wrench(Ff, tf);
        m_elev_force.set_wrench(Fe, te);

        // slew the command; the servo itself closes the loop each substep
        update_command(dt);
        m_servo.set_command(m_cmd);

        m_system.process(dt, SUBSTEPS);
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        const Vector2d o = m_fluid->origin();
        const double vmax = 2.0 * INFLOW;

        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                Vector2d vel;
                m_fluid->velocity_at(Vector2d(wx, wy), &vel,
                                     Fluid::Interp::Cubic);
                val = vel.norm() / vmax;
                const double pert = std::hypot(vel.x() - INFLOW, vel.y());
                const double pa = std::clamp(
                    (pert - PERT_MIN) / (PERT_REF - PERT_MIN), 0.0, 1.0);
                const double dye = std::clamp(
                    m_fluid->density_at(Vector2d(wx, wy), Fluid::Interp::Cubic),
                    0.0, 1.0);
                a = std::max(pa, dye);
            });

        Rendering::draw_spring_damper(r, m_anchor_x.p, m_foil.p);
        Rendering::draw_spring_damper(r, m_anchor_y.p, m_foil.p);
        Rendering::draw_ground_anchor(
            r, m_anchor_x.p,
            {.size = 0.3, .theta = -M_PI / 2, .draw_node = false});
        Rendering::draw_ground_anchor(
            r, m_anchor_y.p, {.size = 0.3, .theta = 0.0, .draw_node = false});

        Rendering::draw_torsion_spring(r, m_foil.p, m_foil.theta,
                                       {.turns = 3, .r_outer = 0.1});

        if (m_mouse.active())
            Rendering::draw_spring(r, m_foil.p, m_mouse.target());

        // rigid tail boom from the foil TE out to the hinge
        Vector2d te, hinge;
        m_foil.local_to_world(Vector2d(m_foil_te, 0.0), &te);
        m_foil.local_to_world(m_foil_anchor, &hinge);
        r->draw_line(te.x(), te.y(), hinge.x(), hinge.y(), 3.0f,
                     Rendering::palette::text_dim());

        const Rendering::Color fg = Rendering::palette::background();
        Rendering::draw_aerofoil(r, m_foil_outline, m_foil.p, m_foil.theta, fg,
                                 fg);
        Rendering::draw_aerofoil(r, m_elev_outline, m_elev.p, m_elev.theta, fg,
                                 fg);
        Rendering::draw_pin_joint(r, hinge, 0.06);

        const double cl = 2.0 * m_last_F.y() / (INFLOW * INFLOW * CHORD);
        const double cd = 2.0 * m_last_F.x() / (INFLOW * INFLOW * CHORD);
        const double defl = (m_elev.theta - m_foil.theta) * 180.0 / M_PI;
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("AEROFOIL + ELEVATOR", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Fx: %+.3f  Fy: %+.3f",
                 m_last_F.x(), m_last_F.y());
        hud.line(Rendering::palette::accent3(), "Cd: %+.2f   Cl: %+.2f", cd,
                 cl);
        hud.line(Rendering::palette::text(), "AoA: %+.1f deg   w: %+.2f",
                 m_foil.theta * 180.0 / M_PI, m_foil.v_theta);
        hud.line(Rendering::palette::accent2(),
                 "elevator: %+.1f deg (cmd %+.1f)", defl, m_cmd * 180.0 / M_PI);
        hud.line(Rendering::palette::text(), "solver: %s",
                 m_use_mac ? "MAC" : "Stam");
        hud.separator();
        hud.small_text("Up/Down elevator  Left-drag  [M] solver  [R] reset",
                       Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        m_fluid->clear_sources();

        if (r->is_key_pressed(Rendering::keys::R)) {
            initialize();
            return;
        }

        if (r->is_key_pressed(Rendering::keys::M)) {
            m_use_mac = !m_use_mac;
            m_fluid = m_use_mac ? (Fluid::FluidSolver *)&m_mac
                                : (Fluid::FluidSolver *)&m_stam;
            m_fluid->clear();
            m_fluid->set_channel(INFLOW);
        }

        m_key_up = r->is_key_down(Rendering::keys::Up);
        m_key_down = r->is_key_down(Rendering::keys::Down);

        int mx, my;
        r->get_mouse_pos(&mx, &my);
        double wx, wy;
        r->screen_to_world(mx, my, &wx, &wy);
        const Vector2d w(wx, wy);

        const bool down = r->is_mouse_button_down(Rendering::mouse::Left);
        if (r->is_mouse_button_pressed(Rendering::mouse::Left))
            m_dragging = near_foil(w);

        if (down && m_dragging) {
            m_mouse.set_active(true);
            m_mouse.set_target(w);
        } else {
            m_dragging = false;
            m_mouse.set_active(false);
        }

        if (down && !m_dragging) {
            int ci, cj;
            if (m_fluid->world_to_cell(w, &ci, &cj))
                for (int dj = -BRUSH; dj <= BRUSH; ++dj)
                    for (int di = -BRUSH; di <= BRUSH; ++di)
                        m_fluid->add_density_source(ci + di, cj + dj,
                                                    DENS_RATE);
        }
    }

  private:
    // spring-centered stick: hold to deflect, release to self-center
    void update_command(double dt) {
        if (m_key_up)
            m_cmd += CMD_RATE * dt;
        else if (m_key_down)
            m_cmd -= CMD_RATE * dt;
        else {
            const double step = CMD_RETURN * dt;
            if (m_cmd > step)
                m_cmd -= step;
            else if (m_cmd < -step)
                m_cmd += step;
            else
                m_cmd = 0.0;
        }
        m_cmd = std::clamp(m_cmd, -CMD_MAX, CMD_MAX);
    }

    void build_foil() {
        std::vector<Vector2d> mask =
            Fluid::naca_points(FOIL_CODE, CHORD, PANELS);
        std::vector<Vector2d> draw =
            Fluid::naca_points(FOIL_CODE, CHORD, RENDER_PANELS);

        const Vector2d com = Fluid::polygon_centroid(draw);
        double te_x = -1e30;
        for (auto &p : mask)
            p -= com;
        for (auto &p : draw) {
            p -= com;
            te_x = std::max(te_x, p.x());
        }

        m_foil_boundary.set_local_sdf(Fluid::polygon_sdf(mask));
        m_foil_outline = draw;
        m_foil.I = Fluid::polygon_inertia(draw, MASS);
        m_foil_te = te_x;
        m_foil_anchor = Vector2d(te_x + HINGE_GAP, 0.0);
    }

    void build_elevator() {
        std::vector<Vector2d> mask =
            Fluid::naca_points(ELEV_CODE, ELEV_CHORD, PANELS);
        std::vector<Vector2d> draw =
            Fluid::naca_points(ELEV_CODE, ELEV_CHORD, RENDER_PANELS);

        const Vector2d com = Fluid::polygon_centroid(draw);
        double le_x = 1e30;
        for (auto &p : mask)
            p -= com;
        for (auto &p : draw) {
            p -= com;
            le_x = std::min(le_x, p.x());
        }

        m_elev_boundary.set_local_sdf(Fluid::polygon_sdf(mask));
        m_elev_outline = draw;
        m_elev.I = Fluid::polygon_inertia(draw, ELEV_MASS);
        m_elev_anchor = Vector2d(le_x, 0.0);
    }

    // put the elevator COM so its LE anchor meets the foil's hinge point
    void place_elevator() {
        Vector2d hinge;
        m_foil.local_to_world(m_foil_anchor, &hinge);
        const double c = std::cos(m_elev.theta), s = std::sin(m_elev.theta);
        const Vector2d off(c * m_elev_anchor.x() - s * m_elev_anchor.y(),
                           s * m_elev_anchor.x() + c * m_elev_anchor.y());
        m_elev.p = hinge - off;
    }

    bool near_foil(const Vector2d &w) const {
        return (w - m_foil.p).norm() < CHORD * 0.7;
    }

    Fluid::StableFluidSolver m_stam{
        (unsigned)ROWS, (unsigned)COLS,
        CELL,           /*visc*/ 0.0,
        /*diff*/ 0.0,   Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};
    Fluid::MACFluidSolver m_mac{
        (size_t)ROWS, (size_t)COLS,
        CELL,         /*visc*/ 0.0,
        /*diff*/ 0.0, Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};
    Fluid::FluidSolver *m_fluid = &m_stam;
    bool m_use_mac = false;

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_foil, m_elev, m_anchor_x, m_anchor_y;
    Solver::Spring m_spring_x, m_spring_y;
    Solver::TorsionSpring m_torsion;
    Solver::LinkConstraint m_hinge;
    Control::ElevatorServo m_servo;
    Solver::MouseSpringForceGenerator m_mouse;

    Coupling::FluidWrenchForce m_foil_force{&m_foil};
    Coupling::FluidWrenchForce m_elev_force{&m_elev};
    Coupling::RigidBodyBoundary m_foil_boundary{
        &m_foil, Fluid::naca_sdf(FOIL_CODE, CHORD, PANELS)};
    Coupling::RigidBodyBoundary m_elev_boundary{
        &m_elev, Fluid::naca_sdf(ELEV_CODE, ELEV_CHORD, PANELS)};

    Vector2d m_rest = Vector2d::Zero();
    Vector2d m_last_F = Vector2d::Zero();
    double m_last_tau = 0.0;
    bool m_dragging = false;

    double m_foil_te = 0.0; // foil trailing-edge x, foil frame
    Vector2d m_foil_anchor = Vector2d::Zero(); // hinge point, foil frame
    Vector2d m_elev_anchor = Vector2d::Zero(); // hinge point, elevator frame

    double m_cmd = 0.0; // commanded deflection (rad)
    bool m_key_up = false, m_key_down = false;

    std::vector<Vector2d> m_foil_outline, m_elev_outline;
    Rendering::FieldView m_field;
};

} // namespace manifold::Demo
