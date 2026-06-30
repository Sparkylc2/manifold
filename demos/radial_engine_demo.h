#pragma once

#include <manifold/renderer/body_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/solver/constraints/fixed_position_constraint.h>
#include <manifold/solver/constraints/fixed_rotation_constraint.h>
#include <manifold/solver/constraints/line_constraint.h>
#include <manifold/solver/constraints/link_constraint.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include <array>
#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

// a radial engine: one driven crankshaft, N cylinders arranged in a star, each
// a crank-slider (rod + piston on a radial rail) sharing the single crank pin.
// the crank is driven at constant speed, so the pin orbit is prescribed and
// every cylinder responds independently -> the proven slider-crank, x N.
class RadialEngineDemo : public DemoBase {
  public:
    static constexpr int N = 7;            // cylinders
    static constexpr double RC = 0.32;     // crank pin radius
    static constexpr double ROD = 1.25;    // connecting-rod length
    static constexpr double R_FLY = 0.42;  // flywheel radius (visual)
    static constexpr double PIS_W = 0.30;  // piston block (radial, tangential)
    static constexpr double PIS_H = 0.34;
    static constexpr double OMEGA = 2.6;   // crank speed (rad/s)
    static constexpr int STEPS = 40;

    const char *name() const override { return "Radial Engine"; }
    double default_cam_zoom() const override { return 200.0; }

    void initialize() override {
        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);

        m_fly.reset();
        m_fly.m = 30.0;
        m_fly.I = 0.5 * 30.0 * R_FLY * R_FLY;
        m_fly.p = Vector2d(0, 0);
        m_fly.theta = 0.0;
        m_system.add_body(&m_fly);

        m_fly_pin.set_body(&m_fly);
        m_fly_pin.set_world_position(Vector2d(0, 0));
        m_fly_pin.set_local_position(Vector2d(0, 0));
        m_fly_pin.set_ks(200.0);
        m_fly_pin.set_kd(20.0);
        m_system.add_constraint(&m_fly_pin);

        const Vector2d pin(RC, 0.0); // crank pin at crank angle 0

        for (int k = 0; k < N; k++) {
            const double phi = k * 2.0 * M_PI / N;
            const double cx = std::cos(phi), sy = std::sin(phi);
            // piston radius that closes the rod loop at this crank angle
            const double r =
                RC * cx + std::sqrt(ROD * ROD - RC * RC * sy * sy);
            const Vector2d pis(r * cx, r * sy);

            m_piston[k].reset();
            m_piston[k].m = 2.0;
            m_piston[k].I = 0.5 * 2.0 * PIS_W * PIS_W;
            m_piston[k].p = pis;
            m_piston[k].theta = phi;
            m_system.add_body(&m_piston[k]);

            const Vector2d mid = 0.5 * (pin + pis);
            const Vector2d d = pis - pin;
            m_rod[k].reset();
            m_rod[k].m = 1.0;
            m_rod[k].I = 1.0 * ROD * ROD / 12.0;
            m_rod[k].p = mid;
            m_rod[k].theta = std::atan2(d.y(), d.x());
            m_system.add_body(&m_rod[k]);

            // rod inner end <-> crank pin
            m_rod_crank[k].set_bodies(&m_fly, &m_rod[k]);
            m_rod_crank[k].set_local_pos1(Vector2d(RC, 0));
            m_rod_crank[k].set_local_pos2(Vector2d(-ROD / 2.0, 0));
            m_rod_crank[k].set_ks(120.0);
            m_rod_crank[k].set_kd(12.0);
            m_system.add_constraint(&m_rod_crank[k]);

            // rod outer end <-> piston
            m_rod_pis[k].set_bodies(&m_rod[k], &m_piston[k]);
            m_rod_pis[k].set_local_pos1(Vector2d(ROD / 2.0, 0));
            m_rod_pis[k].set_local_pos2(Vector2d(0, 0));
            m_rod_pis[k].set_ks(120.0);
            m_rod_pis[k].set_kd(12.0);
            m_system.add_constraint(&m_rod_pis[k]);

            // piston slides along its radial cylinder axis
            m_rail[k].set_body(&m_piston[k]);
            m_rail[k].set_line(Vector2d(0, 0), Vector2d(cx, sy));
            m_rail[k].set_local_pos(Vector2d(0, 0));
            m_rail[k].set_ks(120.0);
            m_rail[k].set_kd(12.0);
            m_system.add_constraint(&m_rail[k]);

            // piston doesn't rotate (stays aligned with the cylinder)
            m_pis_rot[k].set_body(&m_piston[k]);
            m_pis_rot[k].set_angle(phi);
            m_pis_rot[k].set_ks(120.0);
            m_pis_rot[k].set_kd(12.0);
            m_system.add_constraint(&m_pis_rot[k]);
        }
    }

    void process(double dt) override {
        m_fly.v_theta = OMEGA; // drive the crankshaft
        m_system.process(dt, STEPS);
    }

    void render(Rendering::Renderer *r) override {
        const auto a1 = Rendering::palette::accent1();
        const auto a2 = Rendering::palette::accent2();
        const auto dim = Rendering::palette::text_dim();
        const auto fg = Rendering::palette::foreground();

        // cylinder housings (static radial casings)
        for (int k = 0; k < N; k++) {
            const double phi = k * 2.0 * M_PI / N;
            const Vector2d dir(std::cos(phi), std::sin(phi));
            const Vector2d a = (ROD - RC) * dir;
            const Vector2d b = (ROD + RC + 0.25) * dir;
            const Vector2d mid = 0.5 * (a + b);
            Rendering::draw_body_bar(
                r, mid.x(), mid.y(), (b - a).norm(), PIS_H * 1.25, phi,
                {.fill = dim, .show_center = false, .show_shadow = false});
        }

        // rods + pistons
        for (int k = 0; k < N; k++) {
            Rendering::draw_body_bar(r, m_rod[k].p, ROD, 0.07, m_rod[k].theta,
                                     {.show_shadow = true});
            Rendering::draw_body_block(r, m_piston[k].p, PIS_W, PIS_H,
                                       m_piston[k].theta,
                                       {.show_shadow = true});
        }

        // flywheel hub + crank pin + web
        Rendering::draw_arc(r, 0, 0, R_FLY, 0, 2.0 * M_PI, 1.5f, a1, 48);
        Rendering::draw_body_disk(r, m_fly.p, R_FLY * 0.5, m_fly.theta,
                                  {.show_shadow = true});
        Vector2d pin;
        m_fly.local_to_world(Vector2d(RC, 0), &pin);
        r->draw_line(0, 0, pin.x(), pin.y(), 2.0f, a2);
        r->draw_circle(pin.x(), pin.y(), 0.06, a2);

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("RADIAL ENGINE", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "cylinders: %d", N);
        hud.line(Rendering::palette::text(), "crank: %.0f rad/s", OMEGA);
        hud.line(Rendering::palette::accent3(), "bodies: %d   joints: %d",
                 1 + 2 * N, 1 + 4 * N);
        hud.separator();
        hud.small_text("[R] reset   [H] home", Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
    }

  private:
    Solver::GenericRigidBodySystem m_system;
    Solver::GaussianEliminationSLESolver m_sle;
    Solver::RK4ODESolver m_rk4;

    Solver::RigidBody m_fly;
    Solver::FixedPositionConstraint m_fly_pin;

    std::array<Solver::RigidBody, N> m_rod, m_piston;
    std::array<Solver::LinkConstraint, N> m_rod_crank, m_rod_pis;
    std::array<Solver::LineConstraint, N> m_rail;
    std::array<Solver::FixedRotationConstraint, N> m_pis_rot;
};

} // namespace manifold::Demo
