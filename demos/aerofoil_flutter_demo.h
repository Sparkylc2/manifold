#pragma once

#include <manifold/coupling/fluid_wrench_force.h>
#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/fluid/mac_fluid_solver.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
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

// aeroelastic flutter: a NACA aerofoil on two plunge springs + a torsional
// (pitch) spring in a channel flow, two-way coupled through volume
// penalization. visuals follow the "showcase" flutter cell (speed + dye).
class AerofoilFlutterDemo : public DemoBase {
  public:
    static constexpr int COLS = 250;
    static constexpr int ROWS = 100;
    static constexpr double CELL = 0.055;
    static constexpr int SS = 2;

    static constexpr double INFLOW = 12.0;
    static constexpr double CHORD = 2;         // base chord (world units)
    static constexpr double SCALE = 1.0;       // foil size multiplier
    static constexpr double C = CHORD * SCALE; // effective chord, used below
    static constexpr double THICK = 0.12;      // thickness as a fraction of C
    static constexpr int PANELS = 32; // crude SDF panels for penalization
    static constexpr int RENDER_PANELS = 64; // smooth outline
    // [N] cycles these NACA 4-digit profiles (12 == symmetric 0012)
    static constexpr int CODES[6] = {2412, 4412, 2415, 6409, 12, 15};

    static constexpr double MASS = 0.5;
    static constexpr double SPRING_K = 40.0; // plunge stiffness
    static constexpr double ANCHOR_L = 1.6;
    static constexpr double TORSION_K = 6.5; // pitch stiffness
    static constexpr double TORSION_C = 0.0;
    static constexpr double AOA0 = -0.5; // initial angle of attack (rad)
    static constexpr int SUBSTEPS = 8;

    static constexpr int BRUSH = 2;          // dye splat radius (cells)
    static constexpr double DENS_RATE = 240; // dye per second under the cursor

    // alpha ramp: cells whose flow deviates from the free stream light up
    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int FADE_PX = 16;

    const char *name() const override { return "Aerofoil Flutter"; }

    void initialize() override {
        // configure both solvers so [M] can hot-swap between them live
        m_stam.clear();
        m_stam.set_channel(INFLOW);
        m_mac.clear();
        m_mac.set_channel(INFLOW);
        m_mac.set_smoke(true); // MAC needs its dye/advection path on
        m_fluid = m_use_mac ? (Fluid::FluidSolver *)&m_mac
                            : (Fluid::FluidSolver *)&m_stam;

        const Vector2d o = m_fluid->origin();
        m_rest = o + Vector2d(0.32 * COLS * CELL, 0.5 * ROWS * CELL);

        m_foil.reset();
        m_foil.m = MASS;
        m_foil.p = m_rest;
        m_foil.theta = AOA0;
        build_foil(); // sets the SDF, outline, and the polygon inertia
                      // (m_foil.I)

        m_anchor_x.reset();
        m_anchor_x.p = m_rest + Vector2d(-ANCHOR_L, 0.0);
        m_anchor_y.reset();
        m_anchor_y.p = m_rest + Vector2d(0.0, -ANCHOR_L);

        m_system.reset();
        m_system.initialize(&m_sle, &m_rk4);
        m_system.add_body(&m_foil);

        for (auto *sp : {&m_spring_x, &m_spring_y}) {
            sp->set_local_pos1(Vector2d::Zero());
            sp->set_local_pos2(Vector2d::Zero());
            sp->set_rest_length(ANCHOR_L);
            sp->set_ks(SPRING_K);
            sp->set_kd(0.0);
        }
        m_spring_x.set_bodies(&m_anchor_x, &m_foil);
        m_spring_y.set_bodies(&m_anchor_y, &m_foil);
        m_system.add_force_generator(&m_spring_x);
        m_system.add_force_generator(&m_spring_y);

        // torsional spring restores the pitch toward zero AoA
        m_torsion.set_body(&m_foil);
        m_torsion.set_rest_angle(AOA0 / 2.0);
        m_torsion.set_ks(TORSION_K);
        m_torsion.set_kd(TORSION_C);
        m_system.add_force_generator(&m_torsion);

        m_mouse.set_body(&m_foil);
        m_mouse.set_local(Vector2d::Zero());
        m_mouse.set_ks(10.0);
        m_mouse.set_kd(5.0);
        m_mouse.set_active(false);
        m_system.add_force_generator(&m_mouse);

        m_system.add_force_generator(&m_fluid_force);
        // both solvers see the foil so either can be the active one
        m_stam.add_boundary(&m_boundary);
        m_mac.add_boundary(&m_boundary);

        m_field.init(COLS, ROWS,
                     {.supersample = SS,
                      .edge_fade_px = FADE_PX,
                      .gamma = 0.29,
                      .colorbar = true},
                     Rendering::speed_ramp());
        m_field.set_scale(0.0, 2.0 * INFLOW, "speed");
    }

    void process(double dt) override {
        // thin dye streaks upstream so the wake reads clearly (sources are
        // cleared in on_input, which runs first each frame)
        for (int k = 1; k <= 6; ++k) {
            int cj = ROWS * k / 7;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = 3; di <= 9; di++)
                    m_fluid->add_density_source(di, cj + dj, 90.0);
        }

        m_fluid->advance(dt);

        Vector2d F;
        double tau;
        m_fluid->wrench_on(m_boundary, m_foil.p, &F, &tau);
        m_last_F = F;
        m_last_tau = tau; // pitching moment about the COM (drives rotation)
        m_fluid_force.set_wrench(F, tau);
        m_system.process(dt, SUBSTEPS);
    }

    void render(Rendering::Renderer *r) override {
        draw_world_grid(r);
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

        // plunge springs + ground anchors
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

        draw_foil(r);

        const double cl = 2.0 * m_last_F.y() / (INFLOW * INFLOW * CHORD);
        const double cd = 2.0 * m_last_F.x() / (INFLOW * INFLOW * CHORD);
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("AEROFOIL FLUTTER", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Fx: %+.3f  Fy: %+.3f",
                 m_last_F.x(), m_last_F.y());
        hud.line(Rendering::palette::accent3(), "Cd: %+.2f   Cl: %+.2f", cd,
                 cl);
        hud.line(Rendering::palette::text(), "AoA: %+.1f deg   w: %+.2f",
                 m_foil.theta * 180.0 / M_PI, m_foil.v_theta);
        hud.line(Rendering::palette::accent3(), "pitch moment: %+.3f",
                 m_last_tau);
        hud.line(Rendering::palette::text(), "NACA %04d   solver: %s", m_code,
                 m_use_mac ? "MAC" : "Stam");
        hud.separator();
        hud.small_text("Left-drag  [N] profile  [M] solver  [R] reset",
                       Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        m_fluid->clear_sources(); // runs before process() each frame

        if (r->is_key_pressed(Rendering::keys::R)) {
            initialize();
            return;
        }

        // cycle the aerofoil profile (SDF mask + render outline both update)
        if (r->is_key_pressed(Rendering::keys::N)) {
            m_code_idx = (m_code_idx + 1) % (int)(sizeof(CODES) / sizeof(int));
            m_code = CODES[m_code_idx];
            build_foil();
        }

        // hot-swap the active solver (Stam <-> MAC); clear the new one's wake
        if (r->is_key_pressed(Rendering::keys::M)) {
            m_use_mac = !m_use_mac;
            m_fluid = m_use_mac ? (Fluid::FluidSolver *)&m_mac
                                : (Fluid::FluidSolver *)&m_stam;
            m_fluid->clear();
            m_fluid->set_channel(INFLOW);
        }

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

        // pressing over open flow (not on the foil) injects dye
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
    void build_foil() {
        std::vector<Vector2d> mask = Fluid::naca_points(m_code, CHORD, PANELS);
        std::vector<Vector2d> draw =
            Fluid::naca_points(m_code, CHORD, RENDER_PANELS);

        const Vector2d com = Fluid::polygon_centroid(draw);
        for (auto &p : mask)
            p -= com;
        for (auto &p : draw)
            p -= com;

        m_boundary.set_local_sdf(Fluid::polygon_sdf(mask));
        m_outline = draw;
        m_foil.I = Fluid::polygon_inertia(draw, MASS);
    }

    // grab if the cursor is within ~one chord of the foil COM
    bool near_foil(const Vector2d &w) const {
        return (w - m_foil.p).norm() < CHORD * 0.7;
    }

    static void draw_world_grid(Rendering::Renderer *r) {
        double lw, tw, rw, bw;
        r->screen_to_world(0, 0, &lw, &tw);
        r->screen_to_world(r->screen_width(), r->screen_height(), &rw, &bw);
        const auto lc = Rendering::palette::grid_line();
        const auto ac = Rendering::palette::grid_axis();
        const Color rlc{lc.r, lc.g, lc.b, lc.a}, rac{ac.r, ac.g, ac.b, ac.a};
        const double sp = 1.0;
        for (double gx = std::floor(lw / sp) * sp; gx <= rw; gx += sp) {
            const bool ax = std::fabs(gx) < sp * 0.01;
            int x0, y0, x1, y1;
            r->world_to_screen(gx, bw, &x0, &y0);
            r->world_to_screen(gx, tw, &x1, &y1);
            DrawLineEx({(float)x0, (float)y0}, {(float)x1, (float)y1},
                       ax ? 2.0f : 1.0f, ax ? rac : rlc);
        }
        for (double gy = std::floor(bw / sp) * sp; gy <= tw; gy += sp) {
            const bool ax = std::fabs(gy) < sp * 0.01;
            int x0, y0, x1, y1;
            r->world_to_screen(lw, gy, &x0, &y0);
            r->world_to_screen(rw, gy, &x1, &y1);
            DrawLineEx({(float)x0, (float)y0}, {(float)x1, (float)y1},
                       ax ? 2.0f : 1.0f, ax ? rac : rlc);
        }
    }

    void draw_foil(Rendering::Renderer *r) {
        const size_t n = m_outline.size();
        if (n < 3)
            return;
        const double c = std::cos(m_foil.theta), s = std::sin(m_foil.theta);

        std::vector<Vector2d> world(n);
        std::vector<Vector2> scr(n);
        Vector2 centre{0.0f, 0.0f};
        for (size_t i = 0; i < n; i++) {
            const Vector2d &a = m_outline[i];
            world[i] = m_foil.p +
                       Vector2d(c * a.x() - s * a.y(), s * a.x() + c * a.y());
            int sx, sy;
            r->world_to_screen(world[i].x(), world[i].y(), &sx, &sy);
            scr[i] = Vector2{(float)sx, (float)sy};
            centre.x += scr[i].x;
            centre.y += scr[i].y;
        }
        centre.x /= (float)n;
        centre.y /= (float)n;

        const Rendering::Color fg = Rendering::palette::background();
        const Color fill{(unsigned char)(fg.r), (unsigned char)(fg.g),
                         (unsigned char)(fg.b), 255};
        for (size_t i = 0; i < n; i++) {
            const Vector2 a = scr[i], b = scr[(i + 1) % n];
            DrawTriangle(centre, a, b, fill);
            DrawTriangle(centre, b, a, fill);
        }

        // the fan triangles get sub-pixel thin at the sharp trailing edge and
        // miss pixels there; stroke the perimeter in the fill colour to plug
        // those gaps before the outline goes on top
        for (size_t i = 0; i < n; i++)
            DrawLineEx(scr[i], scr[(i + 1) % n], 2.5f, fill);

        for (size_t i = 0; i < n; i++)
            r->draw_line(world[i].x(), world[i].y(), world[(i + 1) % n].x(),
                         world[(i + 1) % n].y(), 2.0f, fg);
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

    Solver::RigidBody m_foil, m_anchor_x, m_anchor_y;
    Solver::Spring m_spring_x, m_spring_y;
    Solver::TorsionSpring m_torsion;
    Solver::MouseSpringForceGenerator m_mouse;

    Coupling::FluidWrenchForce m_fluid_force{&m_foil};
    Coupling::RigidBodyBoundary m_boundary{
        &m_foil, Fluid::naca_sdf(2412, CHORD, PANELS)};

    Vector2d m_rest = Vector2d::Zero();
    Vector2d m_last_F = Vector2d::Zero();
    double m_last_tau = 0.0;
    bool m_dragging = false;

    int m_code_idx = 0;
    int m_code = CODES[0]; // current NACA profile

    std::vector<Vector2d> m_outline;
    Rendering::FieldView m_field;
};

} // namespace manifold::Demo
