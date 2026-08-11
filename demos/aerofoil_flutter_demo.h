#pragma once

#include <iostream>
#include <manifold/coupling/fluid_wrench_force.h>
#include <manifold/coupling/rigid_body_boundary.h>
#include <manifold/fluid/mac_fluid_solver.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/aero_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/solver/forces/mouse_spring.h>
#include <manifold/solver/forces/spring.h>
#include <manifold/solver/forces/torsion_spring.h>
#include <manifold/solver/gaussian_elimination_sle_solver.h>
#include <manifold/solver/generic_body_system.h>
#include <manifold/solver/rk4_ode_solver.h>

#include "manifold/fluid/particle_tracers.h"
#include "manifold/renderer/renderer.h"
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
    // Domain stated the way CylinderFlutterCell states it, and with the same
    // numbers: SPAN_* is how much channel exists and therefore how the cell is
    // shaped on screen, RES is only resolution. Matching SPAN_L/SPAN_W is what
    // makes this a drop-in for the cylinder in the reel -- the slot geometry,
    // HEAD, and the aspect shared with the nozzle all fall out of it.
    static constexpr double SPAN_L = 6.65; // along the flow, world units
    static constexpr double SPAN_W = 3.95; // across it

    // measured, 0012, cut-cell MAC at CFL 2, free slip, against thin-aerofoil
    // Cl = 2*pi*sin(a):
    //
    //   RES   chord   Cl@4   Cl@8   Cl@12   ms/frame
    //    20     42    0.34   0.47    0.54       18
    //    28     59    0.38   0.43    0.66       51
    //    36     76    0.35   0.47    0.65       99
    //    44     92    0.37   0.49    0.68      168
    //   theory        0.44   0.87    1.31
    //
    // Cl@4 is flat across the whole range, so circulation is established at 20
    // and refining does not buy more of it -- 42 cells of chord is enough for
    // the trailing edge to set it. What is still short is the slope at angle,
    // which is the channel being 2.8 chords tall and the load being pressure
    // only.
    //
    // (an earlier version of this table showed Cl@4 near zero until RES 36.
    // that was the penalization no-slip smearing a one-cell boundary layer over
    // a foil 5 cells thick, not the grid. see set_no_slip.)
    static constexpr double RES = 20.0; // cells per world unit

    static constexpr double CELL = 1.0 / RES;
    static constexpr int COLS = (int)(SPAN_L * RES);
    static constexpr int ROWS = (int)(SPAN_W * RES);
    static constexpr int SS = 2;

    // MacCormack wants the backtrace inside a cell or two, and the wake three
    // chords back carries about twice the energy at CFL 2 than unsplit
    static constexpr double FLUID_CFL = 2.0;
    static constexpr int MAX_FLUID_SUBSTEPS = 4;

    static constexpr double W = COLS * CELL; // solver-space extents
    static constexpr double H = ROWS * CELL;

    // HEAD existed only to align this inlet with the nozzle's exit plane while
    // the two sat side by side in one row. They are stacked now and the nozzle
    // runs horizontally, so there is nothing left to align to -- and the dead
    // space it reserved above the field was only shrinking the cell. Restore
    // 0.16084 / 0.83916 * W if they ever go back to sharing a row.
    static constexpr double HEAD = 0.0;
    static constexpr double EDGE_FADE = 0.055;

    // slower than the 12.0 the standalone demo ran 60: the reel cell wants a
    // flutter you can follow, and the shed frequency scales with it
    static constexpr double INFLOW = 6.0;
    // The 2.1 -> 2.5 cliff recorded here, where |F|max jumped 630x against a
    // predicted 1.7x, was the added-mass instability: added mass goes as chord
    // squared while the foil's own mass was fixed, so the ratio crossed what a
    // staggered loop can carry and the load ran away rather than scaling. It is
    // carried on the inertia now (see build_foil), so chord is no longer capped
    // by it -- what caps it now is the channel, at roughly SPAN_W / 2.
    static constexpr double CHORD = 2.1;       // base chord (world units)
    static constexpr double SCALE = 1.0;       // foil size multiplier
    static constexpr double C = CHORD * SCALE; // effective chord, used below
    static constexpr double THICK = 0.12;      // thickness as a fraction of C
    static constexpr int PANELS = 160; // sdf panels; rasterized once, so free
    static constexpr int RENDER_PANELS = 64; // smooth outline
    // [N] cycles these NACA 4-digit profiles (12 == symmetric 0012)
    static constexpr int CODES[6] = {2412, 4412, 2415, 6409, 12, 15};

    static constexpr double MASS = 0.5;
    static constexpr double ANCHOR_L = 1.6;
    static constexpr int SUBSTEPS = 8;

    // Undamped, this flutter is divergent, not periodic: measured over 8 s the
    // plunge grew 0.87 -> 2.46 and the pitch 18 -> 48 deg, and it only stops
    // when the foil reaches the wall. A little structural damping turns that
    // into a limit cycle that holds amplitude for as long as the frame runs.
    static constexpr double SPRING_C = 0.2;
    static constexpr double TORSION_C = 0.1;

    // Stiffnesses scale with the chord, because the load does. In 2-D the
    // aerodynamic force goes as the chord and the moment as its square, so a
    // chord raised from 1.5 to 2.5 against fixed springs deflects ~1.7x further
    // in plunge and ~2.8x further in pitch -- enough to swing the foil out of
    // the channel, where the boundary meets the wall BC and the whole thing
    // tumbles. Tying K to CHORD holds the deflection instead of the stiffness.
    static constexpr double C_REF = 2.1; // chord these were tuned at
    static constexpr double SPRING_K = 300.0 / 2 * (CHORD / C_REF);
    static constexpr double TORSION_K =
        150.0 * 3 / 4 * (CHORD / C_REF) * (CHORD / C_REF);

    // NEGATIVE, and that matters: at +8 deg the aerodynamic moment reinforces
    // the pitch instead of opposing it, so the foil winds up to ~21 deg and
    // sags 2 units downstream of centre. At -8 deg it winds only ~2.6 deg and
    // sits centred. The torsion spring rests here too, so the foil holds an
    // incidence and flutters about it rather than about zero.
    static constexpr double AOA0 = -8.0 * M_PI / 180.0;

    // The cylinder's border is a world-space ring of BodyStyle::border_width
    // outside the disk. draw_aerofoil strokes in SCREEN px and centres the
    // stroke on the outline, so matching it means doubling the width and
    // converting through the live transform -- see border_px().
    static constexpr double BORDER_W = 2.0 * 0.02;

    // colour ceiling, as in the cylinder cell: puts the accelerated flow over
    // the suction side at the top of the ramp
    static constexpr double VMAX = 2.7 * INFLOW;

    // MAX_LEN is only a ceiling -- the blob's length comes from how far it
    // actually travels, so it stretches over the suction side and rounds up
    // where the wake stalls. this cell runs 5x the flutter demo's inflow, so
    // the ceiling is correspondingly wider
    static constexpr double TRACER_W = 0.02;
    static constexpr double TRACER_MAX_LEN = 0.30;
    // this cell runs 5x the flutter demo's inflow, so the trail has to span
    // proportionally less time to stay a blob rather than a streak
    static constexpr int TRACER_SUBSTEPS = 6;
    static constexpr int TRACERS = 1000;

    static constexpr int BRUSH = 2;          // dye splat radius (cells)
    static constexpr double DENS_RATE = 240; // dye per second under the cursor

    // alpha ramp: cells whose flow deviates from the free stream light up
    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int FADE_PX = 16;

    static constexpr double ALPHA_PADDING_DIST = 1.5; // solver units

    const char *name() const override { return "Aerofoil Flutter"; }

    // a quarter turn clockwise so the flow runs down the frame, exactly as
    // CylinderFlutterCell does it: the field is initialised with its dims
    // swapped and the sampler maps drawn coords back into solver coords, which
    // keeps the texture axis-aligned for draw_texture's quad
    void set_quarter_turn(bool on) { m_turned = on; }

    Rendering::Colormap speed_ramp() {
        return [](double t) -> Rendering::Color {
            const auto c0 = Rendering::palette::background();
            const auto c1 = Rendering::palette::accent4();
            const auto c2 = Rendering::palette::accent2();
            const auto c3 = Rendering::palette::accent3();
            const auto c4 = Rendering::palette::accent1();
            if (t < 0.2)
                return color_lerp(c0, c1, t / 0.2);
            if (t < 0.5)
                return color_lerp(c1, c2, (t - 0.2) / 0.3);
            if (t < 0.8)
                return color_lerp(c2, c3, (t - 0.5) / 0.3);
            return color_lerp(c3, c4, (t - 0.8) / 0.2);
        };
    }

    void initialize() override {
        const Vector2d o = m_fluid->origin();

        // configure both solvers so [M] can
        // hot-swap between them live
        m_stam.clear();
        m_stam.set_channel(INFLOW);
        m_mac.clear();
        m_mac.set_channel(INFLOW);
        m_mac.set_smoke(true); // MAC needs its dye/advection path on
        m_mac.set_cfl(FLUID_CFL, MAX_FLUID_SUBSTEPS);
        m_fluid = m_use_mac ? (Fluid::FluidSolver *)&m_mac
                            : (Fluid::FluidSolver *)&m_stam;
        setup_tracers();

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
            sp->set_kd(SPRING_C);
        }
        m_spring_x.set_bodies(&m_anchor_x, &m_foil);
        m_spring_y.set_bodies(&m_anchor_y, &m_foil);
        m_system.add_force_generator(&m_spring_x);
        m_system.add_force_generator(&m_spring_y);

        // torsional spring restores the pitch toward the incidence it starts
        // at, so the foil flutters about a held angle of attack
        m_torsion.set_body(&m_foil);
        m_torsion.set_rest_angle(AOA0);
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

        // turned: the drawn field is ROWS wide and COLS tall. proportional edge
        // fade and no colourbar, matching the cylinder cell exactly
        if (m_turned)
            m_field.init(ROWS, COLS,
                         {.supersample = SS,
                          .edge_fade_frac = EDGE_FADE,
                          .gamma = 0.29,
                          .colorbar = false},
                         Rendering::speed_ramp());
        else
            m_field.init(COLS, ROWS,
                         {.supersample = SS,
                          .edge_fade_px = FADE_PX,
                          .gamma = 0.29,
                          .colorbar = true},
                         Rendering::speed_ramp());
        m_field.set_scale(0.0, VMAX, "speed");
    }

    void process(double dt) override {
        // no upstream dye streaks: alpha now comes from the freestream
        // deviation alone, so the wake is drawn by where the flow actually
        // departs from the inflow rather than by where the tracer happened to
        // be seeded. the density term in the sampler goes inert with them
        m_fluid->advance(dt);
        m_tracer_system.update(dt, TRACER_SUBSTEPS);

        Vector2d F;
        double tau;
        m_fluid->wrench_on(m_boundary, m_foil.p, &F, &tau);
        m_last_F = F;

        m_last_tau = tau;
        m_fluid_force.set_wrench(F, tau);
        m_system.process(dt, SUBSTEPS);
    }

    void render(Rendering::Renderer *r) override {
        // if (!m_turned)
        //     draw_grid(r);
        render_cell(r);

        if (m_mouse.active())
            Rendering::draw_spring(r, to_draw(m_foil.p),
                                   to_draw(m_mouse.target()));

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

    double get_alpha(const Vector2d &pos, const double a) {
        const Vector2d o = m_fluid->origin();
        const Vector2d max = {o.x() + W, o.y() + H};

        // distance to the nearest edge on each axis
        const double dx = std::min(pos.x() - o.x(), max.x() - pos.x());
        const double dy = std::min(pos.y() - o.y(), max.y() - pos.y());

        // 0 at the edge, ramps to 1 once we're a full pad deep
        const double fx = std::clamp(dx / ALPHA_PADDING_DIST, 0.0, 1.0);
        const double fy = std::clamp(dy / ALPHA_PADDING_DIST, 0.0, 1.0);

        double pow = 1.5;
        double t1 = fx;
        double t2 = fy;
        return std::clamp(std::pow(t1, pow) * std::pow(t2, pow) * a, 0.0, 1.0);

        // double t = a * fx * fy;
        // return std::clamp(std::pow(t, 1.2), 0.0, 1.0);
    }

    void render_cell(Rendering::Renderer *r) override {
        if (!m_trail_shader)
            m_trail_shader = r->load_shader("assets/shaders/tracer.fs");

        const Vector2d fo =
            m_turned ? Vector2d(-0.5 * H, -W) : m_fluid->origin();

        m_field.render(r, fo.x(), fo.y(), CELL,
                       [this](double wx, double wy, double &val, double &a) {
                           const Vector2d p = to_solver(wx, wy);
                           Vector2d vel;

                           m_fluid->velocity_at(p, &vel, Fluid::Interp::Cubic);
                           val = vel.norm() / VMAX;
                           const double pert =
                               std::hypot(vel.x() - INFLOW, vel.y());

                           double t = (pert - PERT_MIN) / (PERT_REF - PERT_MIN);
                           a = std::clamp(std::pow(t + 0.2, 0.4), 0.0, 1.0);
                           a = get_alpha(p, a);
                       });

        // m_field.render(r, fo.x(), fo.y(), CELL,
        //                [this](double wx, double wy, double &val, double &a) {
        //                    const Vector2d p = to_solver(wx, wy);
        //                    Vector2d vel;
        //                    m_fluid->velocity_at(p, &vel,
        //                    Fluid::Interp::Cubic); val = vel.norm() / VMAX;
        //                    const double pert =
        //                        std::hypot(vel.x() - INFLOW, vel.y());
        //
        //                    double t = (pert - PERT_MIN) / (PERT_REF -
        //                    PERT_MIN); a = std::clamp(t, 0.0, 1.0);
        //                });
        //
        // additive is a glow: it can only push toward white, so on a light
        // theme the trail composites to background and disappears
        const bool dark = Rendering::palette::background().r < 128;

        std::vector<Rendering::Vertex2D> mesh;
        m_tracer_system.build_mesh(mesh, TRACER_W, TRACER_MAX_LEN,
                                   Rendering::palette::background());

        // m_tracer_system.build_mesh(mesh, TRACER_W, TRACER_MAX_LEN,
        //                            Rendering::palette::foreground());
        //
        // the tracers integrate in solver space; everything else in this cell
        // is drawn through to_draw, and under a quarter turn the two differ
        for (auto &vert : mesh) {
            const Vector2d d = to_draw(Vector2d(vert.x, vert.y));
            vert.x = d.x();
            vert.y = d.y();
        }

        r->draw_shaded(
            mesh.empty() ? 0 : m_trail_shader, mesh.data(), (int)mesh.size(),
            dark ? Rendering::Blend::Additive : Rendering::Blend::Alpha);

        const Vector2d c = to_draw(m_foil.p);
        const Vector2d ax = to_draw(m_anchor_x.p);
        const Vector2d ay = to_draw(m_anchor_y.p);

        Rendering::draw_spring_damper(r, ax, c);
        Rendering::draw_spring_damper(r, ay, c);
        // anchors turn with the frame: the streamwise one now points up
        Rendering::draw_ground_anchor(r, ax,
                                      {.size = 0.22,
                                       .theta = m_turned ? M_PI : -M_PI / 2,
                                       .draw_node = false});
        Rendering::draw_ground_anchor(
            r, ay,
            {.size = 0.22,
             .theta = m_turned ? 3.0 * M_PI / 2.0 : 0.0,
             .draw_node = false});

        draw_foil(r);
        // the chord and drawn on top, it reads as the pitch spring at the
        // pivot, and winding it with theta shows the incidence directly
        Rendering::draw_torsion_spring(r, c, to_draw_angle(m_foil.theta),
                                       {.turns = 3,
                                        .r_inner = 0.01 * CHORD,
                                        .r_outer = 0.10 * CHORD,
                                        .segments = 160});
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
            setup_tracers(); // it caches the solver it samples
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
    // holds the solver pointer, so it has to be redone whenever [M] swaps which
    // one is live. seeded and culled against the actual grid: the +-5 default
    // straddles this channel and would strand a chunk of them outside the
    // fluid, where velocity_at clamps and they never convect
    void setup_tracers() {
        const Vector2d o = m_fluid->origin();

        m_tracer_system.position_min = o;
        m_tracer_system.position_max = o + Vector2d(COLS * CELL, ROWS * CELL);
        m_tracer_system.min_bound = m_tracer_system.position_min;
        m_tracer_system.max_bound = m_tracer_system.position_max;

        m_tracer_system.init(m_fluid, TRACERS, 0);
    }

    void build_foil() {
        std::vector<Vector2d> mask = Fluid::naca_points(m_code, CHORD, PANELS);
        std::vector<Vector2d> draw =
            Fluid::naca_points(m_code, CHORD, RENDER_PANELS);

        const Vector2d com = Fluid::polygon_centroid(draw);
        for (auto &p : mask)
            p -= com;
        for (auto &p : draw)
            p -= com;

        // rasterized once here, so PANELS is now free to be as fine as the
        // outline: the per-step cost is a bilerp either way
        m_boundary.set_local_sdf(Fluid::cached_sdf(Fluid::polygon_sdf(mask),
                                                   0.75 * CHORD, 0.5 * CELL));
        m_outline = draw;

        // the foil is lighter than the fluid it displaces, so the added mass
        // is not a small correction here -- it is 7x the body mass, and
        // without it on the inertia the staggered loop diverges in 25 frames
        m_foil.m = MASS + Coupling::added_mass(CHORD);
        m_foil.I =
            Fluid::polygon_inertia(draw, MASS) + Coupling::added_inertia(CHORD);
    }

    // grab if the cursor is within ~one chord of the foil COM
    bool near_foil(const Vector2d &w) const {
        return (w - m_foil.p).norm() < CHORD * 0.7;
    }

    // solver -> drawn, about the field origin, then recentred across the flow.
    // identical to CylinderFlutterCell's, so the two cells frame the same way
    Vector2d to_draw(const Vector2d &p) const {
        if (!m_turned)
            return p;
        const Vector2d q = p - m_fluid->origin();
        return Vector2d(q.y() - 0.5 * H, -q.x());
    }

    Vector2d to_solver(double wx, double wy) const {
        if (!m_turned)
            return Vector2d(wx, wy);
        return m_fluid->origin() + Vector2d(-wy, wx + 0.5 * H);
    }

    // the turn is a -90 deg rotation of the plane, so body angles carry it too
    double to_draw_angle(double th) const {
        return m_turned ? th - M_PI_2 : th;
    }

    // draw_aerofoil strokes in SCREEN px while the cylinder's border is a
    // world-space ring, so the two only match if the width is converted
    // through whatever transform the slot is currently fitting the cell at
    static float border_px(Rendering::Renderer *r) {
        int x0, y0, x1, y1;
        r->world_to_screen(0.0, 0.0, &x0, &y0);
        r->world_to_screen(BORDER_W, 0.0, &x1, &y1);
        return (float)std::max(1, std::abs(x1 - x0));
    }

    void draw_foil(Rendering::Renderer *r) {
        Rendering::draw_aerofoil(
            r, m_outline, to_draw(m_foil.p), to_draw_angle(m_foil.theta),
            Rendering::palette::grid_line(), Rendering::palette::foreground(),
            border_px(r));
    }

    Fluid::StableFluidSolver m_stam{
        (unsigned)ROWS, (unsigned)COLS,
        CELL,           /*visc*/ 0.0,
        /*diff*/ 0.0,   Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};
    Fluid::MACFluidSolver m_mac{
        (size_t)ROWS, (size_t)COLS,
        CELL,         /*visc*/ 0.0,
        /*diff*/ 0.0, Vector2d(-COLS * CELL * 0.5, -ROWS * CELL * 0.5)};
    Fluid::FluidSolver *m_fluid = &m_mac;
    bool m_use_mac = true; // cut-cell MAC; [M] still drops back to Stam

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
    bool m_turned = false;

    int m_code_idx = 0;
    int m_code = CODES[0]; // current NACA profile

    std::vector<Vector2d> m_outline;
    Rendering::FieldView m_field;

    Fluid::TracerSystem m_tracer_system;

    unsigned int m_trail_shader = 0;
};

} // namespace manifold::Demo
