#pragma once

#include "manifold/ai/pod.h"
#include "manifold/ai/snapshot_recorder.h"
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;

// Karman vortex street + live POD: as snapshots accumulate, the top modes are
// recomputed and drawn as their own flow fields above the live sim.
class PODKarmanDemo : public DemoBase {
  public:
    static constexpr int COLS = 200;     // N_X
    static constexpr int ROWS = 100;     // N_Y
    static constexpr double CELL = 0.09; // world units per cell
    static constexpr double OX = -COLS * CELL * 0.5;
    static constexpr double OY = -ROWS * CELL * 0.5;

    static constexpr double INFLOW = 3.0;
    static constexpr double VISC = 0.0;
    static constexpr double RADIUS = 0.4;

    // POD controls (capped snapshots so it stays realtime)
    static constexpr int N_MODES = 6;         // panels drawn
    static constexpr int N_CAP = 100;         // rolling snapshot window
    static constexpr uint STRIDE = 3;         // capture every Nth frame
    static constexpr double TRANSIENT = 2.0;  // s before recording starts
    static constexpr int MIN_SNAPS = 12;      // wait for this many first
    static constexpr int RECOMPUTE_EVERY = 60; // frames between POD solves

    const char *name() const override { return "Karman POD"; }

    double default_cam_y() const override { return 1.5; }
    double default_cam_zoom() const override { return 40.0; }

    void initialize() override {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);

        const Vector2d o = m_fluid.origin();
        // cylinder ~30% downstream, nudged off-axis to trigger shedding
        m_center = o + Vector2d(0.30 * COLS * CELL, 0.5 * ROWS * CELL + 0.04);
        m_fluid.set_circle_obstacle(m_center, RADIUS);

        m_field.init(COLS, ROWS,
                     {.supersample = 2, .gamma = 0.29, .colorbar = true},
                     Rendering::speed_ramp());
        m_field.set_scale(0.0, 2.0 * INFLOW, "speed");

        m_recorder = AI::SnapshotRecorder(COLS, ROWS, CELL, o, STRIDE,
                                          TRANSIENT, N_CAP);
        m_have_pod = false;
        m_num_shown = 0;
        m_time = 0.0;
        m_frame = 0;

        for (auto &mv : m_mode_views) {
            mv.init(COLS, ROWS,
                    {.supersample = 1, .gamma = 0.5, .colorbar = false},
                    Rendering::speed_ramp());
            mv.set_scale(0.0, 1.0, "mode");
        }
    }

    void process(double dt) override {
        m_fluid.advance(dt);
        m_time += dt;
        m_recorder.maybe_capture(m_fluid, m_time);
        m_frame++;

        if (m_recorder.history().size() >= MIN_SNAPS &&
            m_frame % RECOMPUTE_EVERY == 0) {
            recompute_pod();
        }
    }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW; // colormap range

        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                val = m_fluid.speed_at(Vector2d(wx, wy), Fluid::Interp::Cubic) /
                      vmax;
                a = 1.0;
            });

        r->draw_circle(m_center.x(), m_center.y(), RADIUS,
                       Rendering::palette::foreground());

        draw_mode_panels(r, o);
        draw_hud(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
    }

  private:
    // peak per-cell speed of a mode, normalizes its colour range (modes are
    // unit-norm over the whole field, so raw values are tiny)
    static double mode_peak_speed(const VectorXd &m) {
        const int nc = (int)m.size() / 2;
        double peak = 1e-12;
        for (int c = 0; c < nc; c++) {
            const double u = m[c], v = m[nc + c];
            peak = std::max(peak, std::sqrt(u * u + v * v));
        }
        return peak;
    }

    void recompute_pod() {
        m_pod.compute(m_recorder.history().matrix());
        m_num_shown = std::min(N_MODES, m_pod.num_modes());
        for (int k = 0; k < m_num_shown; k++) {
            m_mode_data[k] = m_pod.mode(k);
            m_mode_scale[k] = mode_peak_speed(m_mode_data[k]);
        }
        m_have_pod = true;
    }

    void draw_mode_panels(Rendering::Renderer *r, const Vector2d &o) {
        if (!m_have_pod)
            return;

        constexpr double MCELL = CELL * 0.25;
        const double pw = COLS * MCELL, gap = 0.25;
        const double total_w = N_MODES * pw + (N_MODES - 1) * gap;
        const double start_x = o.x() + (COLS * CELL - total_w) * 0.5;
        const double py = o.y() + ROWS * CELL + gap;
        const double sc = MCELL / CELL;
        constexpr int nc = COLS * ROWS;

        for (int k = 0; k < m_num_shown; k++) {
            const double pox = start_x + k * (pw + gap);
            const double poy = py;
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
                    const double u = m[c], v = m[nc + c];
                    val = std::sqrt(u * u + v * v) / scale;
                    a = 1.0;
                });

            // cylinder outline scaled into the panel
            const Vector2d mc = Vector2d(pox, poy) + (m_center - o) * sc;
            r->draw_circle(mc.x(), mc.y(), RADIUS * sc,
                           Rendering::palette::foreground());
        }
    }

    void draw_hud(Rendering::Renderer *r) {
        const Vector2d f = m_fluid.obstacle_force();

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("KARMAN POD", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Snapshots: %d / %d",
                 m_recorder.history().size(), N_CAP);
        hud.line(Rendering::palette::text(), "Lift Fy: %+.3f", f.y());
        if (m_have_pod) {
            hud.separator();
            for (int k = 0; k < m_num_shown; k++)
                hud.line(Rendering::palette::accent3(), "mode %d   E %.0f%%", k,
                         100.0 * m_pod.energy(k));
        } else {
            hud.small_text("warming up...", Rendering::palette::text_dim());
        }
        hud.separator();
        hud.small_text("[R] reset   [H] home", Rendering::palette::text_dim());
    }

    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS, CELL, VISC, 0.0, Vector2d(OX, OY)};

    Vector2d m_center = Vector2d::Zero();
    Rendering::FieldView m_field;

    AI::SnapshotRecorder m_recorder{COLS, ROWS, CELL, Vector2d(OX, OY),
                                    STRIDE, TRANSIENT, N_CAP};
    AI::POD m_pod;
    std::array<Rendering::FieldView, N_MODES> m_mode_views;
    std::array<VectorXd, N_MODES> m_mode_data;
    std::array<double, N_MODES> m_mode_scale{};

    bool m_have_pod = false;
    int m_num_shown = 0;
    double m_time = 0.0;
    int m_frame = 0;
};

} // namespace manifold::Demo
