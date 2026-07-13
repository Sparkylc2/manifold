#pragma once

// "manifold" showcase / info page. A portrait-friendly title card meant for a
// 9:16 story: a wordmark, a few technical lines, and three decorative sims
// running live in the background (a flutter cell, a slider-crank, a pendulum).
// [P] toggles the portrait strip guides (handled by DemoBase).

#include <manifold/coupling/fluid_wrench_force.h>
#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/theme.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/forces/uniform_gravity.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// --------------------------------------------------------------------------
// flutter cell: a cylinder on two springs in a channel flow, two-way coupled.
// occupies a fixed band near the top of the layout.
// --------------------------------------------------------------------------
class InfoFlutterCell {
  public:
    static constexpr int COLS = 160;
    static constexpr int ROWS = 64;
    static constexpr double CELL = 0.05;
    static constexpr int SS = 2;

    static constexpr double INFLOW = 3.0;
    static constexpr double RADIUS = 0.28;
    static constexpr double MASS = 0.5;
    static constexpr double SPRING_K = 7.5;
    static constexpr double ANCHOR_L = 1.2;
    static constexpr int SUBSTEPS = 8;

    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;

    // band placement (world); origin is the field's bottom-left
    static constexpr double OX = -4.0;
    static constexpr double OY = 1.4;

    void initialize() {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);

        const Vector2d o = m_fluid.origin();
        m_rest = o + Vector2d(0.30 * COLS * CELL, 0.5 * ROWS * CELL);

        m_cyl.reset();
        m_cyl.m = MASS;
        m_cyl.I = 0.5 * MASS * RADIUS * RADIUS;
        m_cyl.p = m_rest + Vector2d(0.0, 0.16); // small offset to seed motion

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

        m_fluid.add_boundary(&m_boundary);

        m_field.init(COLS, ROWS,
                     {.supersample = SS,
                      .edge_fade_px = 10,
                      .gamma = 0.29,
                      .colorbar = false},
                     Rendering::speed_ramp());
        m_field.set_scale(0.0, 2.0 * INFLOW, "speed");
    }

    void process(double dt) {
        m_fluid.clear_sources();
        // thin dye streaks injected upstream so the wake stays visible
        for (int k = 1; k <= 5; ++k) {
            int cj = ROWS * k / 6;
            for (int dj = -1; dj <= 1; ++dj)
                m_fluid.add_density_source(3, cj + dj, 90.0);
        }

        m_fluid.advance(dt);

        const Vector2d F = m_fluid.obstacle_force();
        const double tau = m_fluid.obstacle_torque(m_cyl.p);
        m_fluid_force.set_wrench(F, tau);
        m_system.process(dt, SUBSTEPS);
    }

    void render(Rendering::Renderer *r) {
        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW;

        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                Vector2d vel;
                m_fluid.velocity_at(Vector2d(wx, wy), &vel,
                                    Fluid::Interp::Cubic);
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
            {.size = 0.22, .theta = -M_PI / 2, .draw_node = false});
        Rendering::draw_ground_anchor(
            r, m_anchor_y.p, {.size = 0.22, .theta = 0.0, .draw_node = false});
        Rendering::draw_body_disk(r, m_cyl.p, RADIUS, m_cyl.theta,
                                  {.show_shadow = false});
    }

  private:
    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS, CELL, /*visc*/ 0.0,
        /*diff*/ 0.0,   Vector2d(OX, OY)};

    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_cyl, m_anchor_x, m_anchor_y;
    Solver::Spring m_spring_x, m_spring_y;
    Coupling::FluidWrenchForce m_fluid_force{&m_cyl};
    Coupling::RigidBodyBoundary m_boundary{&m_cyl, Fluid::circle_sdf(RADIUS)};

    Vector2d m_rest = Vector2d::Zero();
    Rendering::FieldView m_field;
};

// --------------------------------------------------------------------------
// slider-crank: a driven flywheel turning a connecting rod and piston on a
// horizontal rail. self-contained, placed around a center C.
// --------------------------------------------------------------------------
class InfoCrank {
  public:
    static constexpr double CX = -0.7, CY = -1.25; // flywheel center
    static constexpr double R = 0.45;              // flywheel radius
    static constexpr double RC = 0.30;             // crank-pin radius
    static constexpr double ROD = 1.25;            // connecting-rod length
    static constexpr double ROD_W = 0.06;
    static constexpr double PISTON_W = 0.42, PISTON_H = 0.30;
    static constexpr double OMEGA = 2.2; // rad/s
    static constexpr int STEPS = 40;

    void initialize() {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_fly.reset();
        m_fly.m = 20.0;
        m_fly.I = 0.5 * 20.0 * R * R;
        m_fly.p = Vector2d(CX, CY);
        m_system.add_body(&m_fly);

        m_rod.reset();
        m_rod.m = 1.0;
        m_rod.I = 1.0 * ROD * ROD / 12.0;
        m_rod.p = Vector2d(CX + RC + ROD / 2.0, CY);
        m_system.add_body(&m_rod);

        m_piston.reset();
        m_piston.m = 3.0;
        m_piston.I = 0.5 * 3.0 * PISTON_W * PISTON_W;
        m_piston.p = Vector2d(CX + RC + ROD, CY);
        m_system.add_body(&m_piston);

        m_fly_pin.set_body(&m_fly);
        m_fly_pin.set_world_position(Vector2d(CX, CY));
        m_fly_pin.set_local_position(Vector2d(0, 0));
        m_fly_pin.set_ks(100.0);
        m_fly_pin.set_kd(10.0);
        m_system.add_constraint(&m_fly_pin);

        m_crank_link.set_bodies(&m_fly, &m_rod);
        m_crank_link.set_local_pos1(Vector2d(RC, 0));
        m_crank_link.set_local_pos2(Vector2d(-ROD / 2.0, 0));
        m_crank_link.set_ks(100.0);
        m_crank_link.set_kd(10.0);
        m_system.add_constraint(&m_crank_link);

        m_rod_link.set_bodies(&m_rod, &m_piston);
        m_rod_link.set_local_pos1(Vector2d(ROD / 2.0, 0));
        m_rod_link.set_local_pos2(Vector2d(0, 0));
        m_rod_link.set_ks(100.0);
        m_rod_link.set_kd(10.0);
        m_system.add_constraint(&m_rod_link);

        m_rail.set_body(&m_piston);
        m_rail.set_line(Vector2d(CX, CY), Vector2d(1, 0));
        m_rail.set_local_pos(Vector2d(0, 0));
        m_rail.set_ks(100.0);
        m_rail.set_kd(10.0);
        m_system.add_constraint(&m_rail);

        m_piston_rot.set_body(&m_piston);
        m_piston_rot.set_angle(0);
        m_piston_rot.set_ks(100.0);
        m_piston_rot.set_kd(10.0);
        m_system.add_constraint(&m_piston_rot);
    }

    void process(double dt) {
        m_fly.v_theta = OMEGA; // drive at constant angular velocity
        m_system.process(dt, STEPS);
    }

    void render(Rendering::Renderer *r) {
        const auto accent1 = Rendering::palette::accent1();
        const auto bg = Rendering::palette::background();

        Rendering::draw_arc(r, CX, CY, R, 0, 2.0 * M_PI, 1.5f, accent1, 48);

        r->draw_line(CX, CY, CX + RC + ROD + 0.5, CY, 1.0f,
                     Rendering::palette::grid_axis());

        Rendering::draw_body_disk(r, m_fly.p, R, m_fly.theta,
                                  {.show_shadow = true});

        Vector2d pin;
        m_fly.local_to_world(Vector2d(RC, 0), &pin);

        Rendering::draw_body_bar(r, m_rod.p, ROD, ROD_W, m_rod.theta,
                                 {.show_shadow = true});

        Rendering::draw_body_block(r, m_piston.p, PISTON_W, PISTON_H, 0.0,
                                   {.show_shadow = true});

        r->draw_circle(pin.x(), pin.y(), 0.05, accent1);
        r->draw_circle(CX, CY, 0.045, bg);
    }

  private:
    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_fly, m_rod, m_piston;
    Solver::FixedPositionConstraint m_fly_pin;
    Solver::LinkConstraint m_crank_link, m_rod_link;
    Solver::LineConstraint m_rail;
    Solver::FixedRotationConstraint m_piston_rot;
};

// --------------------------------------------------------------------------
// pendulum: a single rigid bar swinging from a pinned anchor. No damping, so
// it keeps swinging for a perpetually lively accent.
// --------------------------------------------------------------------------
class InfoPendulum {
  public:
    static constexpr double PX = 0.0, PY = -3.3; // pivot
    static constexpr double LEN = 1.5;
    static constexpr double WIDTH = 0.07;
    static constexpr double GRAVITY = 9.81;
    static constexpr int STEPS = 40;

    void initialize() {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_anchor.reset();
        m_anchor.m = 0.0;
        m_anchor.I = 0.0;
        m_anchor.p = Vector2d(PX, PY);
        m_system.add_body(&m_anchor);

        m_pin.set_body(&m_anchor);
        m_pin.set_world_position(Vector2d(PX, PY));
        m_pin.set_local_position(Vector2d(0, 0));
        m_pin.set_ks(100.0);
        m_pin.set_kd(10.0);
        m_system.add_constraint(&m_pin);

        const double angle = M_PI / 2.0 + M_PI / 3.0; // ~150 deg from +x
        m_bar.reset();
        m_bar.m = 1.0;
        m_bar.I = 1.0 * LEN * LEN / 12.0;
        m_bar.theta = angle;
        m_bar.p = Vector2d(PX - std::cos(angle) * LEN / 2.0,
                           PY - std::sin(angle) * LEN / 2.0);
        m_system.add_body(&m_bar);

        m_link.set_bodies(&m_anchor, &m_bar);
        m_link.set_local_pos1(Vector2d(0, 0));
        m_link.set_local_pos2(Vector2d(LEN / 2.0, 0));
        m_link.set_ks(100.0);
        m_link.set_kd(10.0);
        m_system.add_constraint(&m_link);

        m_gravity.set_gravity(GRAVITY);
        m_system.add_force_generator(&m_gravity);
    }

    void process(double dt) { m_system.process(dt, STEPS); }

    void render(Rendering::Renderer *r) {
        Rendering::draw_body_node(r, m_anchor.p, 0.12,
                                  {.fill = Rendering::palette::accent1()});
        Rendering::draw_body_bar(r, m_bar.p, LEN, WIDTH, m_bar.theta,
                                 {.show_shadow = true});

        Vector2d bob;
        m_bar.local_to_world(Vector2d(-LEN / 2.0, 0), &bob);
        Rendering::draw_body_disk(r, bob, 0.16, 0,
                                  {.fill = Rendering::palette::accent2()});
    }

  private:
    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_anchor, m_bar;
    Solver::FixedPositionConstraint m_pin;
    Solver::LinkConstraint m_link;
    Solver::UniformGravityForceGenerator m_gravity;
};

// --------------------------------------------------------------------------
// the info page itself
// --------------------------------------------------------------------------
class InfoDemo : public DemoBase {
  public:
    const char *name() const override { return "manifold — showcase"; }

    double default_cam_x() const override { return 0.0; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 46.0; }

    void initialize() override {
        m_flutter.initialize();
        m_crank.initialize();
        m_pendulum.initialize();
        m_time = 0.0;
    }

    void process(double dt) override {
        m_time += dt;
        m_flutter.process(dt);
        m_crank.process(dt);
        m_pendulum.process(dt);
    }

    void render(Rendering::Renderer *r) override {
        // live sims (fluid texture lands under the recorded overlays)
        m_flutter.render(r);
        m_crank.render(r);
        m_pendulum.render(r);

        draw_overlay(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
    }

  private:
    void draw_overlay(Rendering::Renderer *r) {
        const int sw = r->screen_width();
        const int sh = r->screen_height();
        const bool port = portrait_mode();
        const int cx =
            port ? (portrait_strip_left(r) + portrait_strip_right(r)) / 2
                 : sw / 2;

        auto fade = [&](Rendering::Color c, double delay,
                        double dur = 0.6) -> Rendering::Color {
            double f = std::clamp((m_time - delay) / dur, 0.0, 1.0);
            c.a = (unsigned char)(c.a * f);
            return c;
        };

        auto ctext = [&](const std::string &s, int y, int fs,
                         Rendering::Color c) {
            Rendering::LayerScope ui(r, Rendering::Layer::UI);
            int w = r->measure_text(s, fs);
            r->draw_text(s, cx - w / 2, y, fs, c);
        };

        const auto fg = Rendering::palette::foreground();
        const auto text = Rendering::palette::text();
        const auto dim = Rendering::palette::text_dim();
        const auto a3 = Rendering::palette::accent3();

        const int head_fs = port ? 64 : 82;
        const int tag_fs = port ? 22 : 26;
        const int feat_fs = port ? 17 : 19;
        const int cap_fs = 14;
        const int foot_fs = 16;

        // wordmark, accent rule, tagline — stacked off the header font size
        const int head_y = (int)(sh * 0.05);
        ctext("manifold", head_y, head_fs, fade(fg, 0.0));

        const int rule_y = head_y + head_fs + 10;
        {
            Rendering::LayerScope ui(r, Rendering::Layer::UI);
            int rule_w = port ? 150 : 200;
            r->draw_screen_line(cx - rule_w / 2, rule_y, cx + rule_w / 2,
                                rule_y, 2.0f, fade(a3, 0.15));
        }
        ctext("a unified multiphysics engine", rule_y + 14, tag_fs,
              fade(a3, 0.25));

        // caption above the technical block (sits just under the flutter band)
        ctext("vortex-induced flutter", (int)(sh * 0.425), cap_fs,
              fade(dim, 0.5));

        // technical detail block
        ctext("constraint dynamics · Lagrange multipliers", (int)(sh * 0.47),
              feat_fs, fade(text, 0.7));
        ctext("conjugate-gradient solve · RK4 integration", (int)(sh * 0.505),
              feat_fs, fade(text, 0.85));
        ctext("Stam stable fluids · two-way coupling", (int)(sh * 0.54),
              feat_fs, fade(text, 1.0));

        // captions next to the lower sims
        ctext("slider-crank linkage", (int)(sh * 0.64), cap_fs, fade(dim, 1.15));
        ctext("rigid-body pendulum", (int)(sh * 0.835), cap_fs, fade(dim, 1.3));

        // footer
        ctext("C++20 · Eigen · raylib", (int)(sh * 0.92), foot_fs,
              fade(dim, 1.45));
    }

    InfoFlutterCell m_flutter;
    InfoCrank m_crank;
    InfoPendulum m_pendulum;
    double m_time = 0.0;
};

} // namespace manifold::Demo
