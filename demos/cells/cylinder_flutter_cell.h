#pragma once

// Cylinder on two springs in a channel flow, two-way coupled — the vortex
// street drives the body and the body's motion feeds back into the flow.
//
// Turned a quarter turn so the flow runs *down* the frame. The rotation is
// done at the source rather than by transforming the renderer: the field is
// initialised with its dimensions swapped and the sample callback maps screen
// coords back into solver coords. That keeps the texture axis-aligned, which
// matters because draw_texture blits an axis-aligned quad — a rotated
// TransformRenderer would fit the field into the wrong rect.
//
// Solver space has the flow along +x. Drawing space is that turned clockwise:
//      (sx, sy) -> (sy, -sx),      inverse (wx, wy) -> (-wy, wx)

#include <manifold/coupling/fluid_wrench_force.h>
#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/showcase_cell.h>
#include <manifold/renderer/theme.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <algorithm>
#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class CylinderFlutterCell : public ShowcaseCell {
  public:
    // Two independent knobs, which is the point of stating the domain this
    // way round. SPAN_* is how much channel exists and therefore how the cell
    // is shaped on screen; RES is grid resolution, i.e. cost and detail, and
    // changes nothing about the framing. Stating COLS/ROWS/CELL directly
    // couples the two — you cannot raise the resolution without also resizing
    // the domain, or widen the channel without also changing the cost.
    static constexpr double SPAN_L = 10.65; // along the flow, world units
    static constexpr double SPAN_W = 5.95;  // across it
    static constexpr double RES = 20.0;     // cells per world unit

    static constexpr double CELL = 1.0 / RES;
    static constexpr int COLS = (int)(SPAN_L * RES);
    static constexpr int ROWS = (int)(SPAN_W * RES);
    static constexpr int SS = 2;

    // The texture fades toward its border, so the *visible* field is narrower
    // than the domain. Proportional, so it survives a resize.
    static constexpr double EDGE_FADE = 0.055;

    static constexpr double INFLOW = 3.0;
    static constexpr double RADIUS = 0.36;
    static constexpr double MASS = 0.5;
    // The fluid force scales with the projected area, so a bigger cylinder
    // pulls harder on the same springs. K is raised in proportion to keep the
    // absolute deflection where InfoFlutterCell had it at R = 0.28, K = 7.5.
    static constexpr double SPRING_K = 7.5 * (RADIUS / 0.28);
    static constexpr double ANCHOR_L = 1.60;
    static constexpr int SUBSTEPS = 8;

    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;

    // Colour ceiling. Flow accelerating around the cylinder reaches ~1.6-2.0x
    // inflow at the shoulders, so a ceiling of 2.0 x INFLOW puts that band at
    // the very top of the ramp — the red rim hugging the body. Lifting the
    // ceiling drops the shoulders into the mid-ramp and leaves it for the
    // genuinely fast wake cores.
    static constexpr double VMAX = 2.7 * INFLOW;

    static constexpr double W = COLS * CELL; // solver-space extents
    static constexpr double H = ROWS * CELL;

    const char *label() const override { return "vortex-induced flutter"; }

    // the street has to shed before the cylinder starts moving at all
    double warmup() const override { return 7.0; }

    // Solver x in [0, W] maps to drawn y in [-W, 0]; solver y in [0, H] maps
    // to drawn x, recentred so the cell sits about its own origin.
    //
    // Two constraints tie this cell to the nozzle beside it:
    //   HEAD  matches the fraction of the nozzle's bounds its body occupies
    //         above the exit (15.5%), so equal slot yf/hf lands the exit level
    //         with this inlet.
    //   SPAN_W is chosen so the two cells share an aspect ratio. Without
    //         that, equal on-screen width would need unequal hf, which breaks
    //         the vertical alignment above.
    static constexpr double HEAD = 0.155 / 0.845 * W;

    Bounds bounds() const override { return {-0.5 * H, -W, 0.5 * H, HEAD}; }

    // No gravity generator is registered, here or in InfoFlutterCell — the
    // only forces on the cylinder are the two springs and the fluid wrench.
    void initialize() override {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);

        const Vector2d o = m_fluid.origin();
        m_rest = o + Vector2d(0.30 * W, 0.5 * H);

        m_cyl.reset();
        m_cyl.m = MASS;
        m_cyl.I = 0.5 * MASS * RADIUS * RADIUS;
        m_cyl.p = m_rest + Vector2d(0.0, 0.16); // seed the asymmetry

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
            sp->set_kd(0.0);
        }
        m_spring_x.set_bodies(&m_anchor_x, &m_cyl);
        m_spring_y.set_bodies(&m_anchor_y, &m_cyl);
        m_system.add_force_generator(&m_spring_x);
        m_system.add_force_generator(&m_spring_y);
        m_system.add_force_generator(&m_fluid_force);

        m_fluid.clear_boundaries();
        m_fluid.add_boundary(&m_boundary);
    }

    void process(double dt) override {
        m_fluid.clear_sources();
        // thin dye streaks upstream so the wake stays legible
        for (int k = 1; k <= 5; ++k) {
            const int cj = ROWS * k / 6;
            for (int dj = -1; dj <= 1; ++dj)
                m_fluid.add_density_source(3, cj + dj, 90.0);
        }

        m_fluid.advance(dt);

        const Vector2d F = m_fluid.obstacle_force();
        const double tau = m_fluid.obstacle_torque(m_cyl.p);
        m_fluid_force.set_wrench(F, tau);
        m_system.process(dt, SUBSTEPS);
    }

    void render(Rendering::Renderer *r) override {
        if (!m_ready) {
            // dimensions swapped: the drawn field is ROWS wide, COLS tall
            m_field.init(ROWS, COLS,
                         {.supersample = SS,
                          .edge_fade_frac = EDGE_FADE,
                          .gamma = 0.29,
                          .colorbar = false},
                         Rendering::speed_ramp());
            m_field.set_scale(0.0, VMAX, "speed");
            m_ready = true;
        }

        const double vmax = VMAX;
        m_field.render(
            r, -0.5 * H, -W, CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                const Vector2d p = to_solver(wx, wy);
                Vector2d vel;
                m_fluid.velocity_at(p, &vel, Fluid::Interp::Cubic);
                val = vel.norm() / vmax;
                const double pert = std::hypot(vel.x() - INFLOW, vel.y());
                const double pa = std::clamp(
                    (pert - PERT_MIN) / (PERT_REF - PERT_MIN), 0.0, 1.0);
                const double dye = std::clamp(
                    m_fluid.density_at(p, Fluid::Interp::Cubic), 0.0, 1.0);
                a = std::max(pa, dye);
            });

        const Vector2d c = to_draw(m_cyl.p);
        const Vector2d ax = to_draw(m_anchor_x.p);
        const Vector2d ay = to_draw(m_anchor_y.p);

        Rendering::draw_spring_damper(r, ax, c);
        Rendering::draw_spring_damper(r, ay, c);
        // anchors turn with the frame: the streamwise one now points up
        Rendering::draw_ground_anchor(
            r, ax, {.size = 0.22, .theta = M_PI, .draw_node = false});
        Rendering::draw_ground_anchor(
            r, ay,
            {.size = 0.22, .theta = 3.0 * M_PI / 2.0, .draw_node = false});
        Rendering::draw_body_disk(r, c, RADIUS, m_cyl.theta,
                                  {.show_shadow = false});
    }

  private:
    // solver -> drawn, about the field origin, then recentred across the flow
    Vector2d to_draw(const Vector2d &p) const {
        const Vector2d q = p - m_fluid.origin();
        return Vector2d(q.y() - 0.5 * H, -q.x());
    }

    Vector2d to_solver(double wx, double wy) const {
        return m_fluid.origin() + Vector2d(-wy, wx + 0.5 * H);
    }

    Fluid::StableFluidSolver m_fluid{(unsigned)ROWS, (unsigned)COLS,
                                     CELL,           /*visc*/ 0.0,
                                     /*diff*/ 0.0,   Vector2d::Zero()};

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_cyl, m_anchor_x, m_anchor_y;
    Solver::Spring m_spring_x, m_spring_y;
    Coupling::FluidWrenchForce m_fluid_force{&m_cyl};
    Coupling::RigidBodyBoundary m_boundary{&m_cyl, Fluid::circle_sdf(RADIUS)};

    Vector2d m_rest = Vector2d::Zero();
    Rendering::FieldView m_field;
    bool m_ready = false;
};

} // namespace manifold::Demo
