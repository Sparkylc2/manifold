#pragma once

#include "manifold/fluid/fluid_solver.h"
#include "manifold/renderer/renderer.h"
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <manifold/ai/conv_autoencoder.h>

#include <manifold/ai/snapshot_recorder.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/equation_cache.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/plot_widget.h>
#include <string>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

class CAEKarmanDemo : public DemoBase {
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
    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;

    // for the snapshot recorder
    static constexpr int N_CAP = 200;
    static constexpr uint STRIDE = 3;
    static constexpr int EPOCHS_PER_FRAME = 20;
    static constexpr int RECON_EVERY = 3;

    // layout
    static constexpr double FGAP = 1.5;

    const char *name() const override { return "Karman CAE"; }
    double default_cam_x() const override { return 9.75; }
    double default_cam_y() const override { return -4.75; }
    double default_cam_zoom() const override { return 30.0; }

    void initialize() override {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);
        const Vector2d o = m_fluid.origin();
        m_center = o + Vector2d(0.30 * W, 0.5 * H + 0.04);
        m_fluid.set_circle_obstacle(m_center, RADIUS);

        m_dx.set_weights({{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}});
        m_dy.set_weights({{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}});

        // channels = 1: the sampler stores speed (nx*ny), so History must match
        m_kernel_recorder =
            AI::SnapshotRecorder(COLS, ROWS, CELL, o, STRIDE, N_CAP, 0.0, 1);
        m_recorder =
            AI::SnapshotRecorder(COLS, ROWS, CELL, o, STRIDE, N_CAP, 0.0, 2);

        AI::SnapshotRecorder::SampleFunc sampler =
            [](const Fluid::FluidSolver &f,
               const AI::SnapshotRecorder &s) -> VectorXd {
            VectorXd state(s.nx() * s.ny());
            const Vector2d o = s.origin();
            const double h = s.cell();
            for (int j = 0; j < s.ny(); j++)
                for (int i = 0; i < s.nx(); i++) {
                    const Vector2d c =
                        o + Vector2d((i + 0.5) * h, (j + 0.5) * h);
                    Vector2d v;
                    f.velocity_at(c, &v);
                    state[i + j * s.nx()] = v.norm();
                }
            return state;
        };

        m_kernel_recorder.set_sample_func(sampler);
        init_field(m_field, true, SS);
        init_field(m_convolution_field, false, SS);
        init_field(m_recon_field, false, 1);

        // valid (zeroed) field before the first capture, so render never reads
        // empty
        m_convolved.resize(COLS, ROWS);
    }

    void process(double dt) override {

        m_fluid.advance(dt);
        m_frame++;

        m_kernel_recorder.maybe_capture(m_fluid);
        m_recorder.maybe_capture(m_fluid);

        if (m_kernel_recorder.history().size() > 0) {
            const VectorXd col = m_kernel_recorder.history().col(
                m_kernel_recorder.history().size() - 1);
            Fluid::Field2D snapshot(COLS, ROWS);
            for (size_t k = 0; k < snapshot.size(); k++) {
                snapshot[k] = col[(int)k];
            }

            AI::convolve(snapshot, m_dx, m_gx);
            AI::convolve(snapshot, m_dy, m_gy);

            m_convolved.resize(COLS, ROWS);
            for (int k = 0; k < m_convolved.size(); k++) {
                m_convolved[k] = std::hypot(m_gx[k], m_gy[k]);
            }
        }

        if (m_training && m_cae_built)
            for (int k = 0; k < EPOCHS_PER_FRAME; k++) {
                m_loss_plot.push(m_cae.train_epoch(m_cfg));
            }

        if (m_cae_built && m_frame % RECON_EVERY == 0)
            update_reconstruction();
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW;

        draw_flow_field(r, o, vmax);
        draw_convolution(r, o, vmax);
        draw_reconstruction(r, o, vmax);

        m_loss_plot.render(r, 12, r->screen_height() - 92, 280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();

        if (r->is_key_pressed(Rendering::keys::Space)) {
            build_cae();
            m_training = !m_training;
        }
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

    static void init_field(Rendering::FieldView &fv, bool bar, int ss) {
        fv.init(COLS, ROWS,
                {.supersample = ss,
                 .edge_fade_px = 18,
                 .gamma = 0.29,
                 .colorbar = bar,
                 .bar_margin = 24},
                Rendering::speed_ramp());
        fv.set_scale(0.0, 2.0 * INFLOW, "speed");
    }

    void build_cae() {
        m_cae = AI::ConvolutionalAutoencoder();
        m_cae.build(2, COLS, ROWS, {16, 32, 64, 128}, m_latent);
        m_cae.set_data(m_kernel_recorder.history().matrix());
        m_cae_built = true;
        m_training = true;
        m_loss_plot.clear();
    }

    void update_reconstruction() {
        const MatrixXd X = m_recorder.sample_state(m_fluid);
        const MatrixXd Z = m_cae.encode(X);
        const VectorXd X_hat = m_cae.decode(X);

        m_recon = X_hat;
    }

    void draw_flow_field(Rendering::Renderer *r, const Vector2d &o,
                         double vmax) {
        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                val = m_fluid.speed_at(Vector2d(wx, wy)) / vmax;
                a = 1.0;
            });

        label(r, o.x() + W, o.y() + H / 1.5, "LIVE FLOW",
              Rendering::palette::accent2());
    }

    void draw_convolution(Rendering::Renderer *r, const Vector2d &o,
                          double vmax) {
        const double rox = o.x();
        const double roy = o.y() + H + FGAP;

        constexpr int nc = COLS * ROWS;
        const double fy = o.y() + H + FGAP;
        m_convolution_field.render(
            r, rox, fy, CELL,
            [this, o, fy, vmax](double wx, double wy, double &val, double &a) {
                const int i = (int)std::floor((wx - o.x()) / CELL);
                const int j = (int)std::floor((wy - fy) / CELL);
                if (i < 0 || i >= COLS || j < 0 || j >= ROWS) {
                    a = 0.0;
                    return;
                }
                val = m_convolved((size_t)i, (size_t)j) / vmax;
                a = 1.0;
            });

        label(r, rox + W, roy + H / 1.5, "GRADIENT CONVOLUTION DISPLAY",
              Rendering::palette::accent2(), 16, 1);
    }

    void draw_reconstruction(Rendering::Renderer *r, const Vector2d &o,
                             double vmax) {
        const double rox = o.x();
        const double roy = o.y() + 2 * H + 2 * FGAP;

        constexpr int nc = COLS * ROWS;
        const double fy = o.y() + 2 * H + 2 * FGAP;

        if (m_recon.size() == 0) {
            label(r, rox + W, roy + H, "RECONSTRUCTION  (warming up)",
                  Rendering::palette::text_dim());
            return;
        }
        m_recon_field.render(r, rox, roy, CELL,
                             [this, vmax, rox, roy](double wx, double wy,
                                                    double &val, double &a) {
                                 const int i = (int)((wx - rox) / CELL);
                                 const int j = (int)((wy - roy) / CELL);
                                 if (i < 0 || i >= COLS || j < 0 || j >= ROWS) {
                                     a = 0.0;
                                     return;
                                 }
                                 const int cc = i + j * COLS;
                                 const double u = m_recon[cc],
                                              v = m_recon[nc + cc];
                                 val = std::hypot(u, v) / vmax;
                                 a = freestream_alpha(u, v);
                             });

        const Vector2d mc = m_center - o + Vector2d(rox, roy);
        r->draw_circle(mc.x(), mc.y(), RADIUS,
                       Rendering::palette::foreground());
        label(r, rox + W, roy + H / 1.5, "CAE RECONSTRUCTION",
              Rendering::palette::accent2(), 16, 1);
        label(r, rox + W, roy + H / 1.5, sfmt("Latent dim %d", m_latent),
              Rendering::palette::accent2(), 16, 0);
        // label(r, rox + W, roy + H / 2,
        //       sfmt("Recon err %.1f%%", 100.0 * m_recon_err),
        //       Rendering::palette::accent3(), 16, 0);
    }
    static double freestream_alpha(double u, double v) {
        const double pert = std::hypot(u - INFLOW, v);
        return std::clamp((pert - PERT_MIN) / (PERT_REF - PERT_MIN), 0.0, 1.0);
    }

    void draw_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("KARMAN CAE", Rendering::palette::accent2());
        hud.separator();
    }

    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS, CELL, VISC, 0.0, Vector2d(OX, OY)};

    int m_latent = 5;
    int m_frame = 0;

    Fluid::Field2D m_gx, m_gy, m_convolved;

    bool m_cae_built, m_training = false;

    VectorXd m_recon;

    Vector2d m_center = Vector2d::Zero();
    Rendering::FieldView m_field, m_convolution_field, m_recon_field;
    Rendering::PlotWidget m_loss_plot;
    AI::SnapshotRecorder m_kernel_recorder;
    AI::SnapshotRecorder m_recorder;
    AI::ConvolutionalAutoencoder m_cae;
    AI::ConvolutionalAutoencoder::TrainConfig m_cfg{32, 1e-2};

    Rendering::EquationCache m_eq;

    AI::Kernel m_dx{3, 3};
    AI::Kernel m_dy{3, 3};
};

} // namespace manifold::Demo
