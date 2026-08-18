#pragma once

#include <iostream>
#include <manifold/coupling/fea_body_boundary.h>
#include <manifold/coupling/fea_fluid_load.h>
#include <manifold/coupling/fea_spring_force.h>
#include <manifold/fea/fea_solver.h>
#include <manifold/fluid/mac_fluid_solver.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/annotation_visuals.h>
#include <manifold/renderer/constraint_visuals.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/interpolation.h>
#include <manifold/renderer/plot_widget.h>

#include "manifold/fluid/particle_tracers.h"
#include "manifold/renderer/theme.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class FeaFlutterDemo : public DemoBase {
  public:
    // grid refinement. cell count goes up and cell size goes down together, so
    // COLS*CELL and ROWS*CELL are invariant: the flow occupies the same 9.60 x
    // 4.80 of world, and every slot number tuned against it still holds
    static constexpr double RES_SCALE = 1.5;
    static constexpr double ALPHA_PADDING_DIST = 0.85 * RES_SCALE;
    static constexpr int COLS = (int)(210 * RES_SCALE);
    static constexpr int ROWS = (int)(80 * RES_SCALE);
    static constexpr double CELL = 0.06 / RES_SCALE;
    static constexpr int SS = 2;

    static constexpr double INFLOW = 3.0;
    static constexpr double W = COLS * CELL; // solver-space extents
    static constexpr double H = ROWS * CELL;

    // beam. rho is high on purpose: a metal plate in air is ~1e3-1e4 denser
    // than the fluid, and the mass ratio is what keeps the staggered coupling
    // stable
    static constexpr double BEAM_W = 0.30;
    static constexpr double BEAM_H = 2.00;
    static constexpr double E_MOD = 200000.0;
    static constexpr double RHO_S = 1000.0;
    static constexpr double DAMP_A = 0.1;
    // NW must stay even: the boundary chain and the tip probe take i = NW/2 as
    // the centreline
    static constexpr int NW = 12; // cells across the width
    static constexpr int NH = 30; // cells along the height

    static constexpr double CYL_R = 0.22; // upstream cylinder, Wake mode

    // von Mises colour ceilings, set to the p99 of the nodal-averaged stress
    // each mode actually reaches -- the ramp spans the data instead of
    // saturating a third of the way in
    static constexpr double VM_MAX = 1700.0;
    // temporal ema for the stress map. this, not a dead band in the ramp, is
    // what keeps low-stress nodes from shimmering
    static constexpr double VM_SMOOTH = 0.10;
    static constexpr int FADE_PX = 16;

    static constexpr double RELAX_TWOWAY = 0.1;

    static constexpr double SPR_BASE_X = 3.00, SPR_BASE_Y = 2.60;
    static constexpr int J_LOW = NH / 3, J_HIGH = 2 * NH / 3;
    static constexpr double SPR_K = 1000, SPR_C = 0;
    // Friction on the vertical rail, per mount. The plate weighs
    // RHO_S*BEAM_W*BEAM_H = 600 and both mounts drag on it, so its plunge
    // velocity decays with tau = m/(2*c) -- about 1.5 s here. Enough to stop a
    // net cross-load walking the mount off the rail, loose enough that the
    // shedding still moves it.
    static constexpr double SPR_SLIDE_C = 200.0;
    // world height of each mount pad. k and c are applied explicitly, so a
    // point mount's step limit scales with the nodal mass and vanishes as the
    // mesh is refined; a fixed-size pad keeps it mesh-independent
    static constexpr double SPR_FOOT = 0.20;
    static constexpr double RELAX_SPRINGS = 0.01;
    static constexpr double SPR_LOAD_SCALE = 30.0;
    static constexpr double SPR_ANCHOR_X = 0.35;

    static constexpr double LABEL_PAD_Y = 0.55; // clears the edge fade
    static constexpr int LABEL_SIZE = 15;
    static constexpr double VM_MAX_SPRINGS = 8000.0;

    enum class Drive { TwoWay, Wake, Springs };

    static constexpr double TRACER_W = 0.085;
    static constexpr double TRACER_MAX_LEN = 0.5;
    static constexpr int TRACERS = 500;

    void set_drive(Drive d) { m_drive = d; }
    Drive drive() const { return m_drive; }

    const char *name() const override { return "FEA Flutter"; }

    void initialize() override {
        m_stam.clear();
        m_stam.set_channel(INFLOW);
        m_stam.set_solid_project(m_solid_bc);
        m_mac.clear();
        m_mac.set_channel(INFLOW);
        // U/h already puts this near CFL 1, so at 2 the split only engages
        // where the flow locally accelerates rather than every frame

        m_mac.set_smoke(true);

        m_fluid = active_fluid();

        // seed and cull against the actual grid. the +-5 default straddles a
        // channel that runs x +-9.07, y +-3.44, so a third of the tracers land
        // outside the fluid where velocity_at clamps and never convect
        {
            const Vector2d o = m_fluid->origin();
            const Vector2d span(COLS * CELL, ROWS * CELL);

            double b = 1;
            m_tracer_system.position_min = o + Vector2d(b, 0.0);
            m_tracer_system.position_max = o + span;
            m_tracer_system.min_bound = o - Vector2d(b, 0.0);
            m_tracer_system.max_bound = o + span;
        }
        // reads position_min/max to build its distributions, so it goes last
        m_tracer_system.init(m_fluid, TRACERS, 0);

        build_beam();

        m_cyl_c = Vector2d(m_base_x - 1.1, 0.85 * BEAM_H);
        m_cylinder = std::make_unique<Cylinder>(&m_cyl_c, CYL_R);

        m_boundary = std::make_unique<Coupling::FeaBodyBoundary>(
            m_body.get(), m_chain, 0.5 * BEAM_W);
        m_load =
            std::make_unique<Coupling::FeaFluidLoad>(m_fluid, m_body.get());
        m_load->set_sample_offset(1.5 * CELL);
        apply_load_tuning();

        rebind_boundaries();

        m_field_ready =
            false; // FieldView needs GL, so its deferred to first render
        m_tip_plot.configure("tip dx", Rendering::palette::accent2(), 900);
        m_tip_plot.clear();
        m_t = 0.0;
    }

    void process(double dt) override {
        dt = 1.0 / 60;
        m_t += dt;

        m_fluid->clear_sources();
        if (m_drive == Drive::Springs) {
            // for (int k = 1; k <= 7; k++) {
            //     const int cj = ROWS * k / 8;
            //     for (int dj = -1; dj <= 1; dj++)
            //         m_fluid->add_density_source(3, cj + dj, 90.0);
            // }
        } else {
            for (int k = 1; k <= 5; k++) {
                const int cj = ROWS * k / 6;
                for (int dj = -1; dj <= 1; dj++)
                    for (int di = 3; di <= 8; di++)
                        m_fluid->add_density_source(di, cj + dj, 80.0);
            }
        }

        m_body->clear_loads();

        if (m_drive != Drive::Wake)
            m_boundary->refresh();

        m_fluid->advance(dt);
        m_load->update();

        // the fluid load and the spring mounts are both explicit, so the
        // structure has to stay inside its own step limit even when the frame
        // is slow. the fluid keeps the full dt -- only the cheap side
        // subdivides
        m_substeps = m_body->substeps_for(dt);
        const double sub_dt = dt / m_substeps;
        for (int k = 0; k < m_substeps; k++) {
            if (k) {
                m_body->clear_loads();
                m_load->reapply();
            }
            if (m_drive == Drive::Springs) {
                m_spring_low.update();
                m_spring_high.update();
            }
            m_body->advance(sub_dt);
        }

        const Vector2d tip = m_body->node_position(m_tip);
        m_tip_dx = tip.x() - m_rest_tip.x();
        if (!std::isfinite(m_tip_dx))
            initialize();
        m_tip_plot.push(m_tip_dx);

        m_vm_peak = 0.0;
        for (int e = 0; e < m_body->element_count(); e++)
            m_vm_peak = std::max(m_vm_peak, m_body->von_mises(e));

        m_tracer_system.update(dt);
    }

    void render(Rendering::Renderer *r) override {

        draw_grid(r);
        render_cell(r);
        render_hud(r);
        render_plots(r, {&m_tip_plot});
    }

    void render_cell(Rendering::Renderer *r) override {
        if (!m_tracer_shader)
            m_tracer_shader = r->load_shader("assets/shaders/tracer.fs");
        if (!m_merge_shader)
            m_merge_shader = r->load_shader("assets/shaders/metaball.fs");

        if (!m_field_ready) {
            m_field.init(COLS, ROWS,
                         {.supersample = SS,
                          .edge_fade_px = FADE_PX,
                          .gamma = m_drive == Drive::Springs ? 0.29 : 0.32,
                          .colorbar = false},
                         Rendering::speed_ramp());
            m_field.set_scale(0.0, 2.0 * INFLOW, "speed");
            m_field_ready = true;
        }
        const Vector2d o = m_fluid->origin();
        const double vmax = 2.0 * INFLOW;

        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                const Vector2d p(wx, wy);
                Vector2d vel;
                m_fluid->velocity_at(p, &vel, Fluid::Interp::Cubic);
                val = vel.norm() / vmax;

                const double pert = std::hypot(vel.x() - INFLOW, vel.y());
                const double pa = std::clamp(pert / (0.6 * INFLOW), 0.0, 1.0);
                const double dye = std::clamp(
                    m_fluid->density_at(p, Fluid::Interp::Cubic), 0.0, 1.0);
                a = std::max(pa, dye);
                a = get_alpha(p, a);
            });
        if (m_drive == Drive::Wake)
            r->draw_circle(m_cyl_c.x(), m_cyl_c.y(), CYL_R,
                           Rendering::palette::foreground());
        if (m_drive == Drive::Springs)
            draw_supports(r);

        // additive is a glow: it can only push toward white, so on a light
        // theme the trail composites to background and vanishes. this one was
        // drawing terracotta onto sand and clamping to (255,255,255)
        const bool dark = Rendering::palette::background().r < 128;

        std::vector<Rendering::Vertex2D> mesh;
        m_tracer_system.build_mesh(mesh, TRACER_W, TRACER_MAX_LEN,
                                   Rendering::palette::background());

        // tracer.fs emits premultiplied alpha; it has to be resolved through
        // the merge pass or the blend applies alpha twice and the blobs go grey
        const auto blend =
            dark ? Rendering::Blend::Additive : Rendering::Blend::Alpha;
        if (!mesh.empty() && m_merge_shader) {
            r->begin_offscreen();
            r->draw_shaded(m_tracer_shader, mesh.data(), (int)mesh.size(),
                           blend);
            r->end_offscreen(m_merge_shader, blend);
        } else {
            r->draw_shaded(mesh.empty() ? 0 : m_tracer_shader, mesh.data(),
                           (int)mesh.size(), blend);
        }

        draw_beam(r);
        draw_speed_label(r);
        draw_stress_label(r);
    }

    void draw_speed_label(Rendering::Renderer *r) const {
        const double top = m_fluid->origin().y() + ROWS * CELL;
        Rendering::LayerScope txt(r, Rendering::Layer::Text);
        int sx, sy;
        r->world_to_screen(SPR_ANCHOR_X, top - LABEL_PAD_Y, &sx, &sy);
        r->draw_text("Speed", sx, sy, LABEL_SIZE,
                     Rendering::palette::text_dim());
    }
    void draw_stress_label(Rendering::Renderer *r) const {
        const double bot = m_fluid->origin().y();
        Rendering::LayerScope txt(r, Rendering::Layer::Text);
        int sx, sy;
        r->world_to_screen(SPR_ANCHOR_X, bot + LABEL_PAD_Y, &sx, &sy);
        r->draw_text("Stress", sx, sy, LABEL_SIZE,
                     Rendering::palette::text_dim());
    }

    void render_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("FEA FLUTTER", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "tip dx: %+.4f   peak vM: %.0f",
                 m_tip_dx, m_vm_peak);
        hud.line(Rendering::palette::accent3(), "drive: %s   solver: %s",
                 drive_name(), m_use_mac ? "MAC" : "Stam");
        hud.line(Rendering::palette::text(), "BC: %s",
                 m_solid_bc ? "solid-project (sealed)" : "penalization (soft)");
        hud.line(Rendering::palette::text(),
                 "E: %.0f   rho_s: %.0f   U: %.1f   fea x%d", E_MOD, RHO_S,
                 INFLOW, m_substeps);
        hud.separator();
        hud.small_text("[Left/Right] drive  [M] solver  [P] BC  [R] reset",
                       Rendering::palette::text_dim());
    }

    const char *drive_name() const {
        switch (m_drive) {
        case Drive::TwoWay:
            return "two-way cantilever (beam sheds)";
        case Drive::Wake:
            return "wake (cylinder drives cantilever)";
        case Drive::Springs:
            return "spring-mounted plate (free-floating)";
        }
        return "";
    }

    double default_cam_x() const override { return 0.5 * COLS * CELL; }
    double default_cam_y() const override { return 0.42 * ROWS * CELL; }
    double default_cam_zoom() const override { return 105.0; }

  protected:
    void on_input(Rendering::Renderer *r) override {
        m_fluid->clear_sources();

        if (r->is_key_pressed(Rendering::keys::R)) {
            initialize();
            return;
        }

        int step = 0;
        if (r->is_key_pressed(Rendering::keys::Right))
            step = 1;
        if (r->is_key_pressed(Rendering::keys::Left))
            step = -1;
        if (step != 0) {
            m_drive = (Drive)(((int)m_drive + step + 3) % 3);
            build_beam();
            m_load =
                std::make_unique<Coupling::FeaFluidLoad>(m_fluid, m_body.get());
            m_load->set_sample_offset(1.5 * CELL);
            apply_load_tuning();
            m_boundary = std::make_unique<Coupling::FeaBodyBoundary>(
                m_body.get(), m_chain, 0.5 * BEAM_W);
            rebind_boundaries();
            m_fluid->clear();
            m_fluid->set_channel(INFLOW);
            m_tip_plot.clear();
        }

        if (r->is_key_pressed(Rendering::keys::M)) {
            m_use_mac = !m_use_mac;
            m_fluid = active_fluid();
            m_load =
                std::make_unique<Coupling::FeaFluidLoad>(m_fluid, m_body.get());
            m_load->set_sample_offset(1.5 * CELL);
            apply_load_tuning();
            m_fluid->clear();
            m_fluid->set_channel(INFLOW);
        }

        if (r->is_key_pressed(Rendering::keys::P)) {
            m_solid_bc = !m_solid_bc;
            m_stam.set_solid_project(m_solid_bc);
            m_stam.clear();
            m_stam.set_channel(INFLOW);
        }
    }

  private:
    class Cylinder : public Fluid::SolidBoundary {
      public:
        Cylinder(const Vector2d *c, double r) : m_c(c), m_r(r) {}
        double signed_distance(const Vector2d &x) const override {
            return (x - *m_c).norm() - m_r;
        }
        void velocity_at(const Vector2d &, Vector2d *v) const override {
            *v = Vector2d::Zero();
        }

      private:
        const Vector2d *m_c;
        double m_r;
    };

    void apply_load_tuning() {
        switch (m_drive) {
        case Drive::TwoWay:
            m_load->set_relax(RELAX_TWOWAY);
            break;
        case Drive::Wake:
            m_load->set_relax(1.0);
            break;
        case Drive::Springs:
            m_load->set_relax(RELAX_SPRINGS);
            m_load->set_scale(SPR_LOAD_SCALE);
            break;
        }
    }

    Fluid::FluidSolver *active_fluid() {
        return m_use_mac ? (Fluid::FluidSolver *)&m_mac
                         : (Fluid::FluidSolver *)&m_stam;
    }

    void rebind_boundaries() {
        for (Fluid::FluidSolver *f :
             {(Fluid::FluidSolver *)&m_stam, (Fluid::FluidSolver *)&m_mac}) {
            f->clear_boundaries();
            if (m_drive == Drive::Wake)
                f->add_boundary(m_cylinder.get());
            else
                f->add_boundary(m_boundary.get());
        }
    }

    void build_beam() {
        const bool spr = m_drive == Drive::Springs;
        m_base_x = spr ? SPR_BASE_X : 0.30 * COLS * CELL;
        const double y0 = spr ? SPR_BASE_Y - 0.5 * BEAM_H : 0.0;

        FEA::Mesh mesh;
        for (int j = 0; j <= NH; j++)
            for (int i = 0; i <= NW; i++)
                mesh.add_node(
                    Vector2d(m_base_x - 0.5 * BEAM_W + BEAM_W * i / NW,
                             y0 + BEAM_H * j / NH));

        auto id = [](int i, int j) { return j * (NW + 1) + i; };
        for (int j = 0; j < NH; j++)
            for (int i = 0; i < NW; i++) {
                mesh.add_tri(id(i, j), id(i + 1, j), id(i + 1, j + 1));
                mesh.add_tri(id(i, j), id(i + 1, j + 1), id(i, j + 1));
            }
        mesh.build_boundary();

        FEA::Material mat;
        mat.E = E_MOD;
        mat.nu = 0.3;
        mat.rho = RHO_S;
        mat.thickness = 1.0;
        mat.alpha = DAMP_A;

        m_body = std::make_unique<FEA::ElasticBody>(mesh, mat);
        m_body->build();
        if (spr) {
            m_node_low = id(0, J_LOW);
            m_node_high = id(0, J_HIGH);

            // the pad spans SPR_FOOT of the upstream face, centred on the
            // nominal mount node. interior nodes of a uniform column all carry
            // the same tributary mass, so equal weights are the mass shares
            auto pad = [&](int j_centre) {
                const double yc = y0 + BEAM_H * j_centre / NH;
                std::vector<int> ns;
                for (int j = 0; j <= NH; j++)
                    if (std::abs(y0 + BEAM_H * j / NH - yc) <= 0.5 * SPR_FOOT)
                        ns.push_back(id(0, j));
                return ns;
            };

            m_spring_low = Coupling::FeaSpringForce(
                mesh.rest(m_node_low), m_body.get(), m_node_low, SPR_K, SPR_C);
            m_spring_high =
                Coupling::FeaSpringForce(mesh.rest(m_node_high), m_body.get(),
                                         m_node_high, SPR_K, SPR_C);
            for (auto *s : {&m_spring_low, &m_spring_high}) {
                const std::vector<int> ns =
                    pad(s == &m_spring_low ? J_LOW : J_HIGH);
                s->set_footprint(ns, std::vector<double>(ns.size(), 1.0));
                // Streamwise only, so the mounts slide up and down their rail.
                // draw_supports() has always drawn these as a horizontal
                // spring off a vertical wall with the attachment tracking the
                // node's current y -- a roller. Springing both axes pinned the
                // plate to its rest height and contradicted that picture.
                s->set_axis(Vector2d::UnitX());
                s->set_rail_friction(SPR_SLIDE_C);
            }
        } else {
            for (int i = 0; i <= NW; i++)
                m_body->set_fixed_node(id(i, 0), mesh.rest(id(i, 0)));
        }

        m_chain.clear();
        for (int j = 0; j <= NH; j++)
            m_chain.push_back(id(NW / 2, j));

        m_tip = id(NW / 2, NH);
        m_rest_tip = mesh.rest(m_tip);
        m_tip_dx = 0.0;
        m_vm_primed = false;
    }

    // ---- beam stress ramp ----
    //
    // Diverging map: compression hue <- base -> tension hue, lerped in Oklab
    // and signed by hydrostatic stress. The neutral axis has to read as a
    // smooth pass-through, so the response is monotone with no dead band -- an
    // uncoloured plateau there shows up as a hard stripe with saturated colour
    // on both sides, which is what the stops-and-dead-band version did.
    //
    // The two hues carry equal weight only if they sit the same Oklab distance
    // from the base (espresso #3A3028). Measured travel, and clearance to the
    // accents that already colour the fluid:
    //
    //                    travel   nearest accent
    //   straw  D4B24A     0.470     0.072 (accent3 ochre)
    //   fern   4E8C3A     0.292     0.093 (accent2 sage)   <- 1.6x too weak
    //   apple  76C244     0.455     0.149 (accent2 sage)   <- matches straw
    //
    // Anything green reaching straw's 0.470 goes neon, so apple is as close as
    // the palette allows. Temporal noise is VM_SMOOTH's job, not the ramp's.
    static constexpr double STRESS_GAMMA = 1.0;
    // The neutral end cannot be the bare foreground. Espresso sits at Oklab
    // L = 0.32 while both hues are up near L = 0.72, so as stress fell to zero
    // lightness and chroma collapsed together and the neutral axis came out a
    // black crater -- L dropped 0.16 in a single column -- rather than a
    // desaturated line. No gamma or toe fixes that, because the darkness is in
    // the endpoint, not the response. Lifting the neutral toward shadow holds
    // lightness roughly flat across the axis, so the pivot reads as a loss of
    // saturation with a gradient running into it. Raise it to brighten the
    // beam's interior, lower it toward 0 to go back to the dark base
    static constexpr double STRESS_NEUTRAL_LIFT = 0.0;

    static Rendering::Color stress_neutral() {
        return Rendering::color_lerp_oklab(Rendering::palette::foreground(),
                                           Rendering::palette::shadow(),
                                           STRESS_NEUTRAL_LIFT);
    }

    // r is the post-response magnitude; sign picks the hue
    static Rendering::Color stress_colour(double r, bool tension) {
        const Rendering::Color hue =
            tension ? Rendering::Color::hex(0xFF7043FF)  // tomato red
                    : Rendering::Color::hex(0x76C244FF); // apple green
        return Rendering::color_lerp_oklab(stress_neutral(), hue,
                                           std::clamp(r, 0.0, 1.0));
    }

    // gamma below 1 lifts the mid-range; it also sets how wide the desaturated
    // band at the neutral axis reads. no toe -- crushing the response near zero
    // is what put a flat hole at the axis, and VM_SMOOTH already handles the
    // frame-to-frame wobble that the toe was there to hide
    static double stress_response(double m) {
        return std::pow(std::clamp(m, 0.0, 1.0), STRESS_GAMMA);
    }

    void draw_supports(Rendering::Renderer *r) {
        for (int node : {m_node_low, m_node_high}) {
            const Vector2d p = m_body->node_position(node);
            const Vector2d a(SPR_ANCHOR_X, p.y());
            Rendering::draw_spring_damper(r, a, p);
            Rendering::draw_ground_anchor(
                r, a, {.size = 0.22, .theta = -M_PI / 2, .draw_node = false});
        }
    }

    void draw_beam(Rendering::Renderer *r) {
        const FEA::Mesh &mesh = m_body->mesh();
        const int nn = mesh.node_count();

        m_nodal_vm.assign(nn, 0.0);
        m_nodal_cnt.assign(nn, 0.0);
        for (int e = 0; e < m_body->element_count(); e++) {
            const double vm = m_body->von_mises_signed(e);
            const FEA::Tri &t = mesh.tri(e);
            for (int k = 0; k < 3; k++) {
                m_nodal_vm[t.n[k]] += vm;
                m_nodal_cnt[t.n[k]] += 1.0;
            }
        }
        for (int i = 0; i < nn; i++)
            if (m_nodal_cnt[i] > 0.0)
                m_nodal_vm[i] /= m_nodal_cnt[i];

        // low-stress nodes (the tip) jitter frame to frame as the beam
        // vibrates. holds a running average so the colour there is steady
        if ((int)m_nodal_smooth.size() != nn || !m_vm_primed) {
            m_nodal_smooth = m_nodal_vm;
            m_vm_primed = true;
        } else {
            for (int i = 0; i < nn; i++)
                m_nodal_smooth[i] +=
                    VM_SMOOTH * (m_nodal_vm[i] - m_nodal_smooth[i]);
        }

        const double vm_max =
            m_drive == Drive::Springs ? VM_MAX_SPRINGS : VM_MAX;
        // magnitude drives the ramp; sign picks the mid hue (yellow/green)
        auto colour = [&](double vm) {
            return stress_colour(stress_response(std::abs(vm) / vm_max),
                                 vm >= 0.0);
        };

        // one gouraud triangle per element. adjacent elements share edge nodes
        // with identical colours, so the whole surface is continuous
        for (int e = 0; e < m_body->element_count(); e++) {
            const FEA::Tri &t = mesh.tri(e);
            const Vector2d p0 = m_body->node_position(t.n[0]);
            const Vector2d p1 = m_body->node_position(t.n[1]);
            const Vector2d p2 = m_body->node_position(t.n[2]);
            r->draw_triangle_gradient(
                p0.x(), p0.y(), colour(m_nodal_smooth[t.n[0]]), p1.x(), p1.y(),
                colour(m_nodal_smooth[t.n[1]]), p2.x(), p2.y(),
                colour(m_nodal_smooth[t.n[2]]));
        }
    }

    double get_alpha(const Vector2d &pos, const double a) {
        const Vector2d o = m_fluid->origin();
        const Vector2d max = {o.x() + W, o.y() + H};

        // distance to the nearest edge on each axis
        const double dx = std::min(pos.x() - o.x(), max.x() - pos.x());
        const double dy = std::min(pos.y() - o.y(), max.y() - pos.y());

        // 0 at the edge, ramps to 1 once we're a full pad deep.
        // smootherstep, not pow(t, 1.5): that one still had slope 1.4 arriving
        // at t = 1 and 0 just past it, and the eye reads a slope break as a
        // line. this channel is tinted right out to the walls, so there is no
        // texture to hide it behind. zero 1st AND 2nd derivative at both ends
        auto fade = [](double t) {
            t = std::clamp(t, 0.0, 1.0);
            return t * t * t * (t * (6.0 * t - 15.0) + 10.0);
        };
        const double fx = fade(dx / ALPHA_PADDING_DIST);
        const double fy = fade(dy / ALPHA_PADDING_DIST);
        return std::clamp(fx * fy * a, 0.0, 1.0);
    }

    Vector2d to_solver(double wx, double wy) const { return Vector2d(wx, wy); }

    Fluid::StableFluidSolver m_stam{ROWS,   COLS, CELL,
                                    1.0e-6, 0.0,  Vector2d::Zero()};
    Fluid::MACFluidSolver m_mac{ROWS,   COLS, CELL,
                                1.0e-6, 0.0,  Vector2d::Zero()};
    Fluid::FluidSolver *m_fluid = nullptr;
    bool m_use_mac = false; // cut-cell MAC; [M] still drops back to Stam
    bool m_solid_bc = true;

    std::unique_ptr<FEA::ElasticBody> m_body;
    std::unique_ptr<Coupling::FeaBodyBoundary> m_boundary;
    std::unique_ptr<Coupling::FeaFluidLoad> m_load;
    std::unique_ptr<Cylinder> m_cylinder;
    Coupling::FeaSpringForce m_spring_low, m_spring_high;
    int m_node_low = 0, m_node_high = 0;

    std::vector<int> m_chain;
    Drive m_drive = Drive::TwoWay;

    double m_base_x = 0.0;
    int m_tip = 0;
    Vector2d m_rest_tip = Vector2d::Zero();
    Vector2d m_cyl_c = Vector2d::Zero();

    int m_substeps = 1;
    double m_tip_dx = 0.0;
    double m_vm_peak = 0.0;
    double m_t = 0.0;

    std::vector<double> m_nodal_vm;
    std::vector<double> m_nodal_cnt;
    std::vector<double> m_nodal_smooth;
    bool m_vm_primed = false;

    unsigned int m_tracer_shader = 0;
    unsigned int m_merge_shader = 0;
    Fluid::TracerSystem m_tracer_system;

    Rendering::FieldView m_field;
    bool m_field_ready = false;
    Rendering::PlotWidget m_tip_plot;
};

} // namespace manifold::Demo
