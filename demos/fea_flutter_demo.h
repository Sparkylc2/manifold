#pragma once

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
    static constexpr double RES_SCALE = 1.0;
    static constexpr int COLS = (int)(160 * RES_SCALE);
    static constexpr int ROWS = (int)(80 * RES_SCALE);
    static constexpr double CELL = 0.06 / RES_SCALE;
    static constexpr int SS = 2;

    static constexpr double INFLOW = 3.0;

    // beam. rho is high on purpose: a metal plate in air is ~1e3-1e4 denser
    // than the fluid, and the mass ratio is what keeps the staggered coupling
    // stable
    static constexpr double BEAM_W = 0.30;
    static constexpr double BEAM_H = 2.00;
    static constexpr double E_MOD = 60000.0;
    static constexpr double RHO_S = 800.0;
    static constexpr double DAMP_A = 0.08;
    static constexpr int NW = 2;  // cells across the width
    static constexpr int NH = 20; // cells along the height

    static constexpr double CYL_R = 0.22; // upstream cylinder, Wake mode
    static constexpr int SUBSTEPS = 1;

    static constexpr double VM_MAX = 450.0;  // von Mises colour ceiling
                                             // (nodal-averaged, lower than raw)
    static constexpr double VM_SMOOTH = 0.2; // temporal ema for the stress map
    static constexpr int FADE_PX = 16;

    static constexpr double RELAX_TWOWAY = 0.25;

    static constexpr double SPR_BASE_X = 3.00, SPR_BASE_Y = 2.60;
    static constexpr int J_LOW = NH / 3, J_HIGH = 2 * NH / 3;
    static constexpr double SPR_K = 1000.0, SPR_C = 40.0;
    static constexpr double RELAX_SPRINGS = 0.01;
    static constexpr double SPR_LOAD_SCALE = 30.0;
    static constexpr double SPR_ANCHOR_X = 0.35;

    static constexpr double LABEL_PAD_Y = 0.55; // clears the edge fade
    static constexpr int LABEL_SIZE = 15;
    static constexpr double VM_MAX_SPRINGS = 9000.0;

    enum class Drive { TwoWay, Wake, Springs };

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
        m_mac.set_cfl(2.0, 2);

        m_mac.set_smoke(true);
        m_fluid = active_fluid();

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
        if (m_drive == Drive::Springs) {
            m_spring_low.update();
            m_spring_high.update();
        }
        m_body->advance(dt);

        const Vector2d tip = m_body->node_position(m_tip);
        m_tip_dx = tip.x() - m_rest_tip.x();
        if (!std::isfinite(m_tip_dx))
            initialize();
        m_tip_plot.push(m_tip_dx);

        m_vm_peak = 0.0;
        for (int e = 0; e < m_body->element_count(); e++)
            m_vm_peak = std::max(m_vm_peak, m_body->von_mises(e));
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        render_cell(r);
        render_hud(r);
        render_plots(r, {&m_tip_plot});
    }

    void render_cell(Rendering::Renderer *r) override {
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
                Vector2d vel;
                m_fluid->velocity_at(Vector2d(wx, wy), &vel,
                                     Fluid::Interp::Cubic);
                val = vel.norm() / vmax;
                const double pert = std::hypot(vel.x() - INFLOW, vel.y());
                const double pa = std::clamp(pert / (0.6 * INFLOW), 0.0, 1.0);
                const double dye = std::clamp(
                    m_fluid->density_at(Vector2d(wx, wy), Fluid::Interp::Cubic),
                    0.0, 1.0);
                a = std::max(pa, dye);
            });

        if (m_drive == Drive::Wake)
            r->draw_circle(m_cyl_c.x(), m_cyl_c.y(), CYL_R,
                           Rendering::palette::foreground());
        if (m_drive == Drive::Springs)
            draw_supports(r);

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
        hud.line(Rendering::palette::text(), "E: %.0f   rho_s: %.0f   U: %.1f",
                 E_MOD, RHO_S, INFLOW);
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
            m_spring_low = Coupling::FeaSpringForce(
                mesh.rest(m_node_low), m_body.get(), m_node_low, SPR_K, SPR_C);
            m_spring_high =
                Coupling::FeaSpringForce(mesh.rest(m_node_high), m_body.get(),
                                         m_node_high, SPR_K, SPR_C);
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
    // Base -> one saturated hue, lerped in Oklab. Both ends have to stay clear
    // of accent1-4, which already colour the fluid the beam sits in; that
    // clash is what made the original ramp unreadable. Measured in Oklab,
    // against the current text_dim base:
    //
    //            travel from base   nearest accent
    //   wine       0.112              0.112 (accent1 terracotta)
    //   moss       0.332              0.204 (accent4)      <- best on both
    //   straw      0.165              0.049 (accent3 ochre)
    //
    // Stops, low -> high, lerped in Oklab between neighbours. Green for the
    // working range, yellow as it loads, red held back for genuinely high
    // stress. Edit both arrays together in stress_colour() to add or move a
    // stop; a single-hue ramp is just two entries.
    //
    // The green has to clear accent2 (dusty sage), which the fluid ramp uses.
    // Fern sits 0.093 from it -- about the same margin the wine red has from
    // accent1, which already reads fine. A darker moss would be safer (0.208)
    // but travels only 0.144 from this dark base, and that flat dark low end is
    // exactly what was wrong before.
    //
    // Lightness runs 0.318 -> 0.579 -> 0.774 -> 0.460, so it brightens into the
    // yellow then drops into the red: peak stress reads as more saturated
    // rather than brighter, which is what makes the red stand out as a warning
    // rather than just as "more".

    // Sensitivity. t is vm/vm_max, which left most of the beam sitting in the
    // dark end. The old smoothstep flattened BOTH ends, squashing the mid-range
    // it needed to show (t=0.2 came out at 0.10). Now a short toe kills the
    // tip flicker and a gamma below 1 lifts everything above it: t=0.2 lands at
    // 0.38, roughly 3.6x more colour over the working range.
    static constexpr double STRESS_TOE = 0.06;
    static constexpr double STRESS_GAMMA = 0.40;
    // Dead band. Stress below this fraction of vm_max stays at bare base, and
    // the ramp is stretched over what is left. This is the knob for HOW MUCH of
    // the beam carries colour; GAMMA is how fast it climbs once past the band.
    // The two pull against each other -- GAMMA at 0.40 lifts hard, so without a
    // band most of the beam sits in the yellow. Raise MIN to push yellow back
    // toward the root, lower it to spread colour further out to the tip.
    static constexpr double STRESS_MIN = 0.15;
    // Where the last stop before red sits, so red starts blending in from here
    // upward. Lower = red appears further down the beam. Was 0.45; at 0.36 a
    // mid-load element carries roughly twice the red it did.
    static constexpr double STRESS_RED_AT = 0.36;

    // TO SWITCH BASE: same, exactly one stress_lo() live. Leaving both
    // uncommented is a redefinition error, which is the point -- it cannot
    // silently end up in a half-changed state.
    static Rendering::Color stress_lo() {
        return Rendering::palette::foreground();
        // return Rendering::palette::text_dim();
    }
    // darker palette base. straw yellow needs this one; moss green does NOT
    // (its travel collapses 0.332 -> 0.099 against a dark base)
    // static Rendering::Color stress_lo() {
    //     return Rendering::palette::foreground();
    // }

    static Rendering::Color stress_colour(double t) {
        t = std::clamp(t, 0.0, 1.0);
        static const Rendering::Color col[] = {
            stress_lo(), // dark base
            stress_lo(), // dark base
            // Rendering::Color::hex(0x4E8C3AFF),  // fern green
            Rendering::Color::hex(0xD4B24AFF), // straw yellow
            Rendering::Color::hex(0x8C3A3AFF), // wine red
        };

        // todo: we will need to edit this later when doing the high res ones
        // just to make sure the colouring looks nice
        static constexpr double at[] = {0.00, 0.28, STRESS_RED_AT, 1.00};
        constexpr int n = (int)(sizeof(at) / sizeof(at[0]));

        for (int i = 0; i + 1 < n; i++)
            if (t <= at[i + 1])
                return Rendering::color_lerp_oklab(
                    col[i], col[i + 1], (t - at[i]) / (at[i + 1] - at[i]));
        return col[n - 1];
    }

    // toe: flat slope near zero so the unloaded tip does not flicker.
    // gamma: lifts everything above it into the visible part of the ramp
    static double stress_response(double t) {
        t = std::clamp((t - STRESS_MIN) / (1.0 - STRESS_MIN), 0.0, 1.0);
        const double toe = std::clamp(t / STRESS_TOE, 0.0, 1.0);
        return std::pow(t, STRESS_GAMMA) * (toe * toe * (3.0 - 2.0 * toe));
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
            const double vm = m_body->von_mises(e);
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
        auto colour = [&](double vm) {
            return stress_colour(stress_response(vm / vm_max));
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
    Fluid::StableFluidSolver m_stam{ROWS,   COLS, CELL,
                                    1.0e-6, 0.0,  Vector2d::Zero()};
    Fluid::MACFluidSolver m_mac{ROWS,   COLS, CELL,
                                1.0e-6, 0.0,  Vector2d::Zero()};
    Fluid::FluidSolver *m_fluid = nullptr;
    bool m_use_mac = true; // cut-cell MAC; [M] still drops back to Stam
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

    double m_tip_dx = 0.0;
    double m_vm_peak = 0.0;
    double m_t = 0.0;

    std::vector<double> m_nodal_vm;
    std::vector<double> m_nodal_cnt;
    std::vector<double> m_nodal_smooth;
    bool m_vm_primed = false;

    Rendering::FieldView m_field;
    bool m_field_ready = false;
    Rendering::PlotWidget m_tip_plot;
};

} // namespace manifold::Demo
