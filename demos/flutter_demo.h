#pragma once

#include <manifold/coupling/fluid_wrench_force.h>
#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/solver/forces/mouse_spring.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include "manifold/renderer/body_visuals.h"
#include "manifold/renderer/theme.h"
#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class FlutterDemo : public DemoBase {
  public:
    static constexpr int COLS = 330;
    static constexpr int ROWS = 125;
    static constexpr double CELL = 0.055;
    static constexpr int SS = 2;

    static constexpr double INFLOW = 3.0;
    static constexpr double VISC = 0.0;
    static constexpr double RADIUS = 0.3;

    static constexpr double MASS = 0.5;
    static constexpr double SPRING_K = 7.5;
    static constexpr double SPRING_KD = 0;
    static constexpr double ANCHOR_L = 1.6;
    static constexpr double FORCE_SCALE = 1;
    static constexpr int SUBSTEPS = 8;

    static constexpr double DENS_RATE = 120;
    static constexpr int BRUSH = 2;

    // visibility: cells whose velocity deviates from the inflow (u=INFLOW, v=0)
    // by more than PERT_MIN begin to draw, fully opaque at PERT_REF
    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int FADE_PX = 18; // border fade width px

    const char *name() const override { return "Cylinder Flutter"; }

    void initialize() override {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);

        const Vector2d o = m_fluid.origin();
        m_rest = o + Vector2d(0.30 * COLS * CELL, 0.5 * ROWS * CELL);

        m_cyl.reset();
        m_cyl.m = MASS;
        m_cyl.I = 0.5 * MASS * RADIUS * RADIUS;
        m_cyl.p = m_rest;

        m_anchor_x.reset();
        m_anchor_x.p = m_rest + Vector2d(-ANCHOR_L, 0.0);
        m_anchor_y.reset();
        m_anchor_y.p = m_rest + Vector2d(0.0, -ANCHOR_L);

        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);
        m_system.add_body(&m_cyl);

        for (auto *sp : {&m_spring_x, &m_spring_y}) {
            sp->set_local_pos1(Vector2d::Zero());
            sp->set_local_pos2(Vector2d::Zero());
            sp->set_rest_length(ANCHOR_L);
            sp->set_ks(SPRING_K);
            sp->set_kd(SPRING_KD);
        }
        m_spring_x.set_bodies(&m_anchor_x, &m_cyl);
        m_spring_y.set_bodies(&m_anchor_y, &m_cyl);
        m_system.add_force_generator(&m_spring_x);
        m_system.add_force_generator(&m_spring_y);

        m_mouse.set_body(&m_cyl);
        m_mouse.set_local(Vector2d::Zero());
        m_mouse.set_ks(10.0);
        m_mouse.set_kd(5.0);
        m_mouse.set_active(false);
        m_system.add_force_generator(&m_mouse);

        m_system.add_force_generator(&m_fluid_force);

        m_fluid.add_boundary(&m_boundary);

        m_field.init(COLS, ROWS,
                     {.supersample = SS,
                      .edge_fade_px = FADE_PX,
                      .gamma = 0.29,
                      .colorbar = true},
                     Rendering::speed_ramp());
        m_field.set_scale(0.0, 2.0 * INFLOW, "speed");
    }

    void process(double dt) override {
        m_fluid.advance(dt);
        // two-way load: net force + torque (torque now emerges from the
        // per-cell no-slip penalization, not a separate rim model)
        const Vector2d F = m_fluid.obstacle_force() * FORCE_SCALE;
        const double tau = m_fluid.obstacle_torque(m_cyl.p) * FORCE_SCALE;
        m_fluid_force.set_wrench(F, tau);
        m_system.process(dt, SUBSTEPS);
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW;

        // colour = speed; alpha lights up where the flow deviates from the free
        // stream (u=INFLOW, v=0), or where mouse dye is present
        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                Vector2d vel;
                m_fluid.velocity_at(Vector2d(wx, wy), &vel, Fluid::Interp::Cubic);
                val = vel.norm() / vmax;
                const double pert = std::hypot(vel.x() - INFLOW, vel.y());
                const double pa = std::clamp(
                    (pert - PERT_MIN) / (PERT_REF - PERT_MIN), 0.0, 1.0);
                const double dye = std::clamp(
                    m_fluid.density_at(Vector2d(wx, wy), Fluid::Interp::Cubic),
                    0.0, 1.0);
                a = std::max(pa, dye);
            });

        Rendering::draw_spring_damper(r, m_anchor_x.p, m_cyl.p);
        Rendering::draw_spring_damper(r, m_anchor_y.p, m_cyl.p);
        Rendering::draw_ground_anchor(
            r, m_anchor_x.p,
            {
                .size = 0.3,
                .theta = -M_PI / 2,
                .draw_node = false,
                .bar = Rendering::palette::foreground(),
                .hatch = Rendering::palette::foreground(),
            });

        Rendering::draw_ground_anchor(
            r, m_anchor_y.p,
            {
                .size = 0.3,
                .theta = 0.0,
                .draw_node = false,
                .bar = Rendering::palette::foreground(),
                .hatch = Rendering::palette::foreground(),
            });

        if (m_mouse.active())
            Rendering::draw_spring(r, m_cyl.p, m_mouse.target());

        Rendering::draw_body_disk(r, m_cyl.p, RADIUS, m_cyl.theta,
                                  {.show_shadow = false});

        const Vector2d f = m_fluid.obstacle_force();
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("CYLINDER FLUTTER", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Fx: %+.3f  Fy: %+.3f", f.x(),
                 f.y());
        hud.line(Rendering::palette::text(), "pos: %+.2f, %+.2f", m_cyl.p.x(),
                 m_cyl.p.y());
        hud.line(Rendering::palette::accent3(), "vel: %+.2f, %+.2f",
                 m_cyl.v.x(), m_cyl.v.y());
        hud.line(Rendering::palette::accent3(), "omega: %+.2f  tau: %+.3f",
                 m_cyl.v_theta, m_fluid.obstacle_torque(m_cyl.p));
        hud.separator();
        hud.small_text("Left-drag to move; empty space = dye;",
                       Rendering::palette::text_dim());
        hud.small_text("[R] reset   [H] home", Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        m_fluid.clear_sources();

        // if (r->is_key_pressed(Rendering::keys::Space))
        //     m_cyl.v_theta += 10.0; // kick a spin to watch rotational drag
        //
        if (r->is_key_pressed(Rendering::keys::R)) {
            initialize();
            return;
        }

        int mx, my;
        r->get_mouse_pos(&mx, &my);
        double wx, wy;
        r->screen_to_world(mx, my, &wx, &wy);
        const Vector2d w(wx, wy);

        if (r->is_mouse_button_pressed(Rendering::mouse::Left))
            m_dragging = (w - m_cyl.p).norm() < RADIUS * 1.4;

        if (r->is_mouse_button_down(Rendering::mouse::Left)) {
            if (m_dragging) {
                m_mouse.set_active(true);
                m_mouse.set_target(w);
            } else {
                int ci, cj;
                if (m_fluid.world_to_cell(w, &ci, &cj))
                    for (int dj = -BRUSH; dj <= BRUSH; ++dj)
                        for (int di = -BRUSH; di <= BRUSH; ++di)
                            m_fluid.add_density_source(ci + di, cj + dj,
                                                       DENS_RATE);
            }
        } else {
            m_dragging = false;
            m_mouse.set_active(false);
        }
    }

  private:
    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS,
        CELL,           VISC,
        /*diff*/ 0.0,   Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_cyl;
    Solver::RigidBody m_anchor_x, m_anchor_y;
    Solver::Spring m_spring_x, m_spring_y;
    Solver::MouseSpringForceGenerator m_mouse;

    Coupling::FluidWrenchForce m_fluid_force{&m_cyl};
    Coupling::RigidBodyBoundary m_boundary{&m_cyl, Fluid::circle_sdf(RADIUS)};

    Vector2d m_rest = Vector2d::Zero();
    bool m_dragging = false;

    Rendering::FieldView m_field;
};

} // namespace manifold::Demo
