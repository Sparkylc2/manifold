#pragma once

#include <manifold/coupling/fea_body_boundary.h>
#include <manifold/coupling/fea_fluid_load.h>
#include <manifold/fea/fea_solver.h>
#include <manifold/fluid/mac_fluid_solver.h>
#include <manifold/fluid/solid_shapes.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/plot_widget.h>

#include "manifold/renderer/theme.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;

class FeaFlutterDemo : public DemoBase {
  public:
    static constexpr int COLS = 160;
    static constexpr int ROWS = 80;
    static constexpr double CELL = 0.06;
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

    // proper no-through BC (cut the solid out of the pressure solve)
    // over-reacts to boundary motion, so the two-way load is under-relaxed to
    // hold off the added-mass instability. one-way needs no relaxation
    static constexpr double RELAX_TWOWAY = 0.25;

    enum class Drive { TwoWay, Wake };

    const char *name() const override { return "FEA Flutter"; }

    void initialize() override {
        m_stam.clear();
        m_stam.set_channel(INFLOW);
        m_stam.set_solid_project(m_solid_bc); // proper no-through BC
        m_mac.clear();
        m_mac.set_channel(INFLOW);

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
        m_load->set_relax(m_drive == Drive::TwoWay ? RELAX_TWOWAY : 1.0);

        rebind_boundaries();

        m_field.init(COLS, ROWS,
                     {.supersample = SS,
                      .edge_fade_px = FADE_PX,
                      .gamma = 0.32,
                      .colorbar = true},
                     Rendering::speed_ramp());
        m_field.set_scale(0.0, 2.0 * INFLOW, "speed");

        m_tip_plot.configure("tip dx", Rendering::palette::accent2(), 900);
        m_tip_plot.clear();
        m_t = 0.0;
    }

    void process(double dt) override {
        m_t += dt;

        // dye upstream so the wake reads
        for (int k = 1; k <= 5; k++) {
            const int cj = ROWS * k / 6;
            for (int dj = -1; dj <= 1; dj++)
                for (int di = 3; di <= 8; di++)
                    m_fluid->add_density_source(di, cj + dj, 80.0);
        }

        m_body->clear_loads();

        // snapshot the deformed shape before the fluid reads the boundary
        if (m_drive == Drive::TwoWay)
            m_boundary->refresh();

        m_fluid->advance(dt);
        m_load->update();
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

        draw_beam(r);

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("FEA FLUTTER", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "tip dx: %+.4f   peak vM: %.0f",
                 m_tip_dx, m_vm_peak);
        hud.line(Rendering::palette::accent3(), "drive: %s   solver: %s",
                 m_drive == Drive::TwoWay ? "two-way (beam sheds)"
                                          : "wake (cylinder drives)",
                 m_use_mac ? "MAC" : "Stam");
        hud.line(Rendering::palette::text(), "BC: %s",
                 m_solid_bc ? "solid-project (sealed)" : "penalization (soft)");
        hud.line(Rendering::palette::text(), "E: %.0f   rho_s: %.0f   U: %.1f",
                 E_MOD, RHO_S, INFLOW);
        hud.separator();
        hud.small_text("[T] drive  [M] solver  [P] BC  [R] reset",
                       Rendering::palette::text_dim());

        render_plots(r, {&m_tip_plot});
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

        if (r->is_key_pressed(Rendering::keys::T)) {
            m_drive = (m_drive == Drive::TwoWay) ? Drive::Wake : Drive::TwoWay;
            build_beam();
            m_load =
                std::make_unique<Coupling::FeaFluidLoad>(m_fluid, m_body.get());
            m_load->set_sample_offset(1.5 * CELL);
            m_load->set_relax(m_drive == Drive::TwoWay ? RELAX_TWOWAY : 1.0);
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
            m_load->set_relax(m_drive == Drive::TwoWay ? RELAX_TWOWAY : 1.0);
            m_fluid->clear();
            m_fluid->set_channel(INFLOW);
        }

        // toggle the proper no-through BC. if two-way ever destabilises, this
        // drops back to soft penalization (leaks a little, but always stable)
        if (r->is_key_pressed(Rendering::keys::P)) {
            m_solid_bc = !m_solid_bc;
            m_stam.set_solid_project(m_solid_bc);
            m_stam.clear();
            m_stam.set_channel(INFLOW);
        }
    }

  private:
    // static circle, only used to drive the beam in Wake mode
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

    Fluid::FluidSolver *active_fluid() {
        return m_use_mac ? (Fluid::FluidSolver *)&m_mac
                         : (Fluid::FluidSolver *)&m_stam;
    }

    // whichever solid the active drive mode wants, on both solvers so [M] can
    // hot-swap without rebuilding
    void rebind_boundaries() {
        for (Fluid::FluidSolver *f :
             {(Fluid::FluidSolver *)&m_stam, (Fluid::FluidSolver *)&m_mac}) {
            f->clear_boundaries();
            if (m_drive == Drive::TwoWay)
                f->add_boundary(m_boundary.get());
            else
                f->add_boundary(m_cylinder.get());
        }
    }

    void build_beam() {
        m_base_x = 0.30 * COLS * CELL;

        FEA::Mesh mesh;
        for (int j = 0; j <= NH; j++)
            for (int i = 0; i <= NW; i++)
                mesh.add_node(
                    Vector2d(m_base_x - 0.5 * BEAM_W + BEAM_W * i / NW,
                             BEAM_H * j / NH));

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
        for (int i = 0; i <= NW; i++)
            m_body->set_fixed_node(id(i, 0), mesh.rest(id(i, 0)));

        m_chain.clear();
        for (int j = 0; j <= NH; j++)
            m_chain.push_back(id(NW / 2, j));

        m_tip = id(NW / 2, NH);
        m_rest_tip = mesh.rest(m_tip);
        m_tip_dx = 0.0;
        m_vm_primed = false; // stress reset, drop the old temporal average
    }

    // dedicated stress ramp. speed_ramp's low end is the theme background
    // (near-white in the tan theme), which makes low-stress regions vanish on
    // the solid. this ramp is a solid navy->red so the tip stays visible
    static Rendering::Color stress_colour(double t) {
        t = std::clamp(t, 0.0, 1.0);
        const double s[5][3] = {{40, 70, 200},
                                {40, 190, 200},
                                {90, 200, 90},
                                {235, 200, 60},
                                {230, 70, 60}};
        const double x = t * 4.0;
        const int i = std::min(3, (int)x);
        const double f = x - i;
        auto ch = [&](int k) {
            return (unsigned char)(s[i][k] + f * (s[i + 1][k] - s[i][k]));
        };
        return Rendering::Color{ch(0), ch(1), ch(2), 235};
    }

    void draw_beam(Rendering::Renderer *r) {
        const FEA::Mesh &mesh = m_body->mesh();
        const int nn = mesh.node_count();

        // CST stress is piecewise-constant and checkerboards between the two
        // tris of a cell. average element values onto the nodes for a smooth
        // field, the standard FEA post-processing step
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
        // vibrates. hold a running average so the colour there is steady
        if ((int)m_nodal_smooth.size() != nn || !m_vm_primed) {
            m_nodal_smooth = m_nodal_vm;
            m_vm_primed = true;
        } else {
            for (int i = 0; i < nn; i++)
                m_nodal_smooth[i] +=
                    VM_SMOOTH * (m_nodal_vm[i] - m_nodal_smooth[i]);
        }

        auto colour = [&](double vm) {
            double t = std::clamp(vm / VM_MAX, 0.0, 1.0);
            // smoothstep: flat slope at both ends, so tiny fluctuations near
            // zero stress barely move the colour (kills tip flicker)
            t = t * t * (3.0 - 2.0 * t);
            return stress_colour(t);
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

        // outline so the deformed shape reads against the field
        for (int i = 0; i < mesh.edge_count(); i++) {
            const FEA::Edge &ed = mesh.edge(i);
            const Vector2d p0 = m_body->node_position(ed.n[0]);
            const Vector2d p1 = m_body->node_position(ed.n[1]);
            r->draw_smooth_line(p0.x(), p0.y(), p1.x(), p1.y(), 0.022,
                                Rendering::palette::shadow());
        }
    }

    Fluid::StableFluidSolver m_stam{ROWS,   COLS, CELL,
                                    1.0e-6, 0.0,  Vector2d::Zero()};
    Fluid::MACFluidSolver m_mac{ROWS,   COLS, CELL,
                                1.0e-6, 0.0,  Vector2d::Zero()};
    Fluid::FluidSolver *m_fluid = nullptr;
    bool m_use_mac = false;
    bool m_solid_bc = true; // proper no-through BC on the Stam solver

    std::unique_ptr<FEA::ElasticBody> m_body;
    std::unique_ptr<Coupling::FeaBodyBoundary> m_boundary;
    std::unique_ptr<Coupling::FeaFluidLoad> m_load;
    std::unique_ptr<Cylinder> m_cylinder;

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
    Rendering::PlotWidget m_tip_plot;
};

} // namespace manifold::Demo
