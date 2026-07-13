#pragma once

#include "manifold/ai/pod.h"
#include "manifold/ai/snapshot_recorder.h"
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/plot_widget.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <future>
#include <string>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

// Karman vortex street + live POD. The SVD runs on a worker thread so the sim
// never stalls; modes swap in when ready. Modes are drawn as their own flow
// fields beside a live rank-r reconstruction and a KE-error trace.
class PODKarmanDemo : public DemoBase {
  public:
    static constexpr int COLS = 200;
    static constexpr int ROWS = 100;
    static constexpr double CELL = 0.09;
    static constexpr int SS = 2;
    static constexpr double OX = -COLS * CELL * 0.5;
    static constexpr double OY = -ROWS * CELL * 0.5;
    static constexpr double W = COLS * CELL;
    static constexpr double H = ROWS * CELL;

    static constexpr double INFLOW = 3.0;
    static constexpr double VISC = 0.0;
    static constexpr double RADIUS = 0.4;

    // flutter-demo look: transparent where flow ~ freestream, opaque in the
    // wake
    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int FADE_PX = 18;

    // POD controls (capped snapshots so it stays realtime)
    static constexpr int N_MODES = 8;
    static constexpr int MODE_COLS = 2;
    static constexpr int N_CAP = 200;
    static constexpr uint STRIDE = 3;
    static constexpr double TRANSIENT = 2.0;
    static constexpr int MIN_SNAPS = 12;
    static constexpr int RECOMPUTE_EVERY = 60;

    // layout (world units)
    static constexpr double GAP = 1.4;
    static constexpr double MCELL = CELL * 0.48;
    static constexpr double MW = COLS * MCELL;
    static constexpr double MH = ROWS * MCELL;
    static constexpr double MGAP = 0.72; // between mode columns
    static constexpr double VGAP = 2.4;  // between mode rows (room for labels)

    const char *name() const override { return "Karman POD"; }

    double default_cam_y() const override { return 5.0; }
    double default_cam_zoom() const override { return 16.0; }

    void initialize() override {
        // drain any in-flight solve before tearing state down
        if (m_pod_job.valid())
            m_pod_job.wait();
        m_job_running = false;

        m_fluid.clear();
        m_fluid.set_channel(INFLOW);

        const Vector2d o = m_fluid.origin();
        m_center = o + Vector2d(0.30 * W, 0.5 * H + 0.04);
        m_fluid.set_circle_obstacle(m_center, RADIUS);

        m_recorder =
            AI::SnapshotRecorder(COLS, ROWS, CELL, o, STRIDE, N_CAP, TRANSIENT);
        m_have_pod = false;
        m_num_shown = 0;
        m_rank = 4;
        m_ke_rel = 0.0;
        m_time = 0.0;
        m_frame = 0;
        m_recon.resize(0);

        init_field(m_field, true);
        init_field(m_recon_field, false);
        for (auto &mv : m_mode_views) {
            mv.init(COLS, ROWS,
                    {.supersample = 1,
                     .edge_fade_px = 6,
                     .gamma = 0.5,
                     .colorbar = false},
                    Rendering::speed_ramp());
            mv.set_scale(0.0, 1.0, "mode");
        }

        m_ke_plot.configure("KE err %", Rendering::palette::accent3(), 600);
        m_ke_plot.clear();
    }

    void process(double dt) override {
        m_fluid.advance(dt); // sim never blocks on the SVD
        m_time += dt;
        m_recorder.maybe_capture(m_fluid, m_time);
        m_frame++;

        // launch a POD solve off-thread (only if none is running)
        if (!m_job_running && m_recorder.history().size() >= MIN_SNAPS &&
            m_frame % RECOMPUTE_EVERY == 0) {
            MatrixXd snap = m_recorder.history().matrix(); // copy, main thread
            m_pod_job =
                std::async(std::launch::async, [snap = std::move(snap)]() {
                    AI::POD p;
                    p.compute(snap);
                    return p;
                });
            m_job_running = true;
        }

        // collect a finished solve, non-blocking
        if (m_job_running && m_pod_job.wait_for(std::chrono::seconds(0)) ==
                                 std::future_status::ready) {
            m_pod = m_pod_job.get();
            cache_modes();
            m_have_pod = true;
            m_job_running = false;
        }

        if (m_have_pod)
            update_reconstruction();
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);

        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW;

        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                Vector2d v;
                m_fluid.velocity_at(Vector2d(wx, wy), &v, Fluid::Interp::Cubic);
                val = v.norm() / vmax;
                a = freestream_alpha(v.x(), v.y());
            });
        r->draw_circle(m_center.x(), m_center.y(), RADIUS,
                       Rendering::palette::foreground());
        label(r, o.x(), o.y() + H, "LIVE FLOW", Rendering::palette::accent2());

        draw_reconstruction(r, o, vmax);
        draw_mode_panels(r, o);
        draw_hud(r);

        // KE trace, bottom-left (clear of the top-right colourbar)
        m_ke_plot.render(r, r->screen_width() - 292, r->screen_height() - 92,
                         280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Up))
            m_rank = std::min(m_rank + 1, N_MODES);
        if (r->is_key_pressed(Rendering::keys::Down))
            m_rank = std::max(m_rank - 1, 1);
    }

  private:
    static std::string sfmt(const char *f, ...) {
        char buf[160];
        va_list a;
        va_start(a, f);
        std::vsnprintf(buf, sizeof buf, f, a);
        va_end(a);
        return std::string(buf);
    }

    static void label(Rendering::Renderer *r, double wx, double wy,
                      const std::string &s, Rendering::Color c, int sz = 16,
                      int line = 0) {
        int sx, sy;
        r->world_to_screen(wx, wy, &sx, &sy);
        r->draw_text(s, sx + 2, sy - 22 - line * 16, sz, c);
    }

    static void init_field(Rendering::FieldView &fv, bool bar) {
        fv.init(COLS, ROWS,
                {.supersample = SS,
                 .edge_fade_px = FADE_PX,
                 .gamma = 0.29,
                 .colorbar = bar,
                 .bar_margin = 24},
                Rendering::speed_ramp());
        fv.set_scale(0.0, 2.0 * INFLOW, "speed");
    }

    static double freestream_alpha(double u, double v) {
        const double pert = std::hypot(u - INFLOW, v);
        return std::clamp((pert - PERT_MIN) / (PERT_REF - PERT_MIN), 0.0, 1.0);
    }

    static double mode_peak_speed(const VectorXd &m) {
        const int nc = (int)m.size() / 2;
        double peak = 1e-12;
        for (int c = 0; c < nc; c++)
            peak = std::max(peak, std::hypot(m[c], m[nc + c]));
        return peak;
    }

    int rank() const { return std::clamp(m_rank, 1, m_pod.num_modes()); }

    void cache_modes() {
        m_num_shown = std::min(N_MODES, m_pod.num_modes());
        for (int k = 0; k < m_num_shown; k++) {
            m_mode_data[k] = m_pod.mode(k);
            m_mode_scale[k] = mode_peak_speed(m_mode_data[k]);
        }
    }

    void update_reconstruction() {
        const VectorXd x = m_recorder.sample_state(m_fluid);
        m_recon = m_pod.reconstruct(x, rank());

        const double area = CELL * CELL;
        const double ke_fluct = 0.5 * (x - m_pod.mean()).squaredNorm() * area;
        const double ke_err = 0.5 * (x - m_recon).squaredNorm() * area;
        m_ke_rel = ke_fluct > 1e-12 ? ke_err / ke_fluct : 0.0;
        m_ke_plot.push(100.0 * m_ke_rel);
    }

    void draw_reconstruction(Rendering::Renderer *r, const Vector2d &o,
                             double vmax) {
        if (!m_have_pod || m_recon.size() == 0)
            return;

        const double rox = o.x();
        const double roy = o.y() - GAP - H;
        constexpr int nc = COLS * ROWS;

        m_recon_field.render(r, rox, roy, CELL,
                             [this, vmax, rox, roy](double wx, double wy,
                                                    double &val, double &a) {
                                 const int i = (int)((wx - rox) / CELL);
                                 const int j = (int)((wy - roy) / CELL);
                                 if (i < 0 || i >= COLS || j < 0 || j >= ROWS) {
                                     a = 0.0;
                                     return;
                                 }
                                 const int c = i + j * COLS;
                                 const double u = m_recon[c],
                                              v = m_recon[nc + c];
                                 val = std::hypot(u, v) / vmax;
                                 a = freestream_alpha(u, v);
                             });

        const Vector2d mc = m_center - o + Vector2d(rox, roy);
        r->draw_circle(mc.x(), mc.y(), RADIUS,
                       Rendering::palette::foreground());

        label(r, rox, roy + H,
              sfmt("RECONSTRUCTION  rank %d/%d", rank(), m_pod.num_modes()),
              Rendering::palette::accent2());
        label(r, rox, roy + H, sfmt("KE error: %.1f%%", 100.0 * m_ke_rel),
              Rendering::palette::accent3(), 16, 1);
    }

    // 2 columns x 3 rows; mode 0 top-left, index increasing left->right, down
    void draw_mode_panels(Rendering::Renderer *r, const Vector2d &o) {
        if (!m_have_pod)
            return;

        const int rows = (N_MODES + MODE_COLS - 1) / MODE_COLS;
        const double rowstep = MH + VGAP;
        const double x0 = o.x() + W + GAP; // to the right of the flow field
        const double block_h = rows * MH + (rows - 1) * VGAP;
        const double top_edge =
            (o.y() - GAP * 0.5) + block_h * 0.5; // centred on the flow column
        const double sc = MCELL / CELL;
        constexpr int nc = COLS * ROWS;

        for (int k = 0; k < m_num_shown; k++) {
            const int col = k % MODE_COLS, row = k / MODE_COLS;
            const double pox = x0 + col * (MW + MGAP);
            const double poy = top_edge - MH - row * rowstep; // mode 0 top
            const double scale = m_mode_scale[k];

            m_mode_views[k].render(
                r, pox, poy, MCELL,
                [this, k, pox, poy, scale](double wx, double wy, double &val,
                                           double &a) {
                    const int i = (int)((wx - pox) / MCELL);
                    const int j = (int)((wy - poy) / MCELL);
                    if (i < 0 || i >= COLS || j < 0 || j >= ROWS) {
                        a = 0.0;
                        return;
                    }
                    const VectorXd &m = m_mode_data[k];
                    const int c = i + j * COLS;
                    val = std::hypot(m[c], m[nc + c]) / scale;
                    a = std::clamp(val, 0.0, 1.0);
                });

            const Vector2d mc = Vector2d(pox, poy) + (m_center - o) * sc;
            r->draw_circle(mc.x(), mc.y(), RADIUS * sc,
                           Rendering::palette::foreground());

            label(r, pox, poy + MH, sfmt("MODE %d", k),
                  Rendering::palette::text());
            label(r, pox, poy + MH,
                  sfmt("E %.1f%%  cum %.0f%%", 100.0 * m_pod.energy(k),
                       100.0 * m_pod.cumulative_energy(k + 1)),
                  Rendering::palette::text_dim(), 14, 1);
        }
    }

    void draw_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("KARMAN POD", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Snapshots: %d / %d",
                 m_recorder.history().size(), N_CAP);
        if (m_have_pod)
            hud.line(Rendering::palette::accent3(), "Modes: %d   Rank: %d",
                     m_num_shown, rank());
        else
            hud.small_text("warming up...", Rendering::palette::text_dim());
        if (m_job_running)
            hud.small_text("solving POD...", Rendering::palette::accent1());
        hud.separator();
        hud.small_text("[Up/Down] rank   [R] reset   [H] home",
                       Rendering::palette::text_dim());
    }

    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS, CELL, VISC, 0.0, Vector2d(OX, OY)};

    Vector2d m_center = Vector2d::Zero();
    Rendering::FieldView m_field;
    Rendering::FieldView m_recon_field;
    Rendering::PlotWidget m_ke_plot;

    AI::SnapshotRecorder m_recorder{COLS,   ROWS,  CELL,     Vector2d(OX, OY),
                                    STRIDE, N_CAP, TRANSIENT};
    AI::POD m_pod;
    std::array<Rendering::FieldView, N_MODES> m_mode_views;
    std::array<VectorXd, N_MODES> m_mode_data;
    std::array<double, N_MODES> m_mode_scale{};
    VectorXd m_recon;

    std::future<AI::POD> m_pod_job;
    bool m_job_running = false;

    bool m_have_pod = false;
    int m_num_shown = 0;
    int m_rank = 4;
    double m_ke_rel = 0.0;
    double m_time = 0.0;
    int m_frame = 0;
};

} // namespace manifold::Demo
