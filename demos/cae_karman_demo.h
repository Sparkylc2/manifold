#pragma once

#include "manifold/ai/conv_autoencoder.h"
#include "manifold/ai/snapshot_recorder.h"
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/plot_widget.h>
#include <manifold/solver/forces/beam_bending.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

class CAEKarmanDemo : public DemoBase {
  public:
    // full-res fluid + live display
    static constexpr int COLS = 256;
    static constexpr int ROWS = 128;
    static constexpr double CELL = 0.09;
    static constexpr int SS = 2;
    static constexpr double OX = -COLS * CELL * 0.5;
    static constexpr double OY = -ROWS * CELL * 0.5;
    static constexpr double W = COLS * CELL;
    static constexpr double H = ROWS * CELL;

    // coarse grid the CAE actually sees (must halve cleanly per conv layer)
    static constexpr int CX = 256;
    static constexpr int CY = 128;
    static constexpr double CCELL = W / CX; // same world footprint as the flow

    static constexpr double INFLOW = 3.0;
    static constexpr double VISC = 0.0;
    static constexpr double RADIUS = 0.4;
    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int FADE_PX = 18;
    static constexpr double EDGE_PAD = 1.2;

    static constexpr int N_CAP = 160;
    static constexpr uint STRIDE = 3;
    static constexpr double TRANSIENT = 4.0;
    static constexpr int MIN_SNAPS = 96;

    static constexpr int RECON_EVERY = 2;

    static constexpr double FGAP = 1.5;

    const char *name() const override { return "Karman CAE"; }
    double default_cam_x() const override { return 9.75; }
    double default_cam_y() const override { return -4.75; }
    double default_cam_zoom() const override { return 30.0; }

    ~CAEKarmanDemo() override { stop_worker(); }

    void initialize() override {
        stop_worker();

        m_fluid.clear();
        m_fluid.set_channel(INFLOW);
        const Vector2d o = m_fluid.origin();
        m_center = o + Vector2d(0.30 * W, 0.5 * H + 0.04);
        m_fluid.set_circle_obstacle(m_center, RADIUS);

        m_recorder =
            AI::SnapshotRecorder(CX, CY, CCELL, o, STRIDE, N_CAP, TRANSIENT, 2);

        m_time = 0.0;
        m_frame = 0;
        m_want_data = true;
        m_recon.resize(0);
        m_latent_vec.resize(0);
        m_loss = m_recon_err = 0.0;
        m_epochs = 0;
        m_have_recon = false;

        init_field(m_field, COLS, ROWS, true, SS);
        init_field(m_recon_field, CX, CY, false, SS);
        m_loss_plot.configure("CAE loss", Rendering::palette::accent3(), 800);
        m_loss_plot.clear();

        start_worker();

        beam.m_L = 10;
        beam.m_E = 1.0;
        beam.m_I = 10.0;

        beam.add_bc(Solver::BCType::Clamped, 0.0);
        beam.add_bc(Solver::BCType::Free, beam.m_L);
        beam.add_point_load(1, beam.m_L);
        beam.prepare_system();
        beam.solve_system();
        std::cout << "Deflection: " << beam.get_deflection(beam.m_L)
                  << std::endl;
    }

    void process(double dt) override {
        m_fluid.advance(dt);
        m_time += dt;
        m_recorder.maybe_capture(m_fluid, m_time);
        m_frame++;

        // hand the worker a frozen training window once enough is captured
        if (m_want_data && m_recorder.history().size() >= MIN_SNAPS) {
            MatrixXd snaps = m_recorder.history().matrix();
            {
                std::lock_guard<std::mutex> lk(m_mtx);
                m_train_snaps = std::move(snaps);
                m_data_dirty = true;
            }
            m_cv.notify_all();
            m_want_data = false;
        }

        // publish the current field for the worker to reconstruct
        if (m_frame % RECON_EVERY == 0) {
            VectorXd live = m_recorder.sample_state(m_fluid);
            std::lock_guard<std::mutex> lk(m_mtx);
            m_live_state = std::move(live);
        }

        // pull the latest reconstruction the worker published
        std::lock_guard<std::mutex> lk(m_mtx);
        if (m_pub_dirty) {
            m_recon = m_pub_recon;
            m_latent_vec = m_pub_latent;
            m_loss = m_pub_loss;
            m_recon_err = m_pub_err;
            m_epochs = m_pub_epochs;
            m_have_recon = m_recon.size() > 0;
            m_loss_plot.push(m_loss);
            m_pub_dirty = false;
        }
    }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW;

        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax, o](double wx, double wy, double &val, double &a) {
                Vector2d v;
                m_fluid.velocity_at(Vector2d(wx, wy), &v,
                                    Fluid::Interp::Linear);
                val = v.norm() / vmax;
                a = freestream_alpha(v.x(), v.y()) *
                    Rendering::window_alpha(wx - o.x(), wy - o.y(), 0.0, 0.0, W,
                                            H, EDGE_PAD);
            });
        r->draw_circle(m_center.x(), m_center.y(), RADIUS,
                       Rendering::palette::foreground());
        label(r, o.x() + W, o.y() + H / 1.5, "LIVE FLOW",
              Rendering::palette::accent2());

        draw_reconstruction(r, o, vmax);
        draw_hud(r);
        m_loss_plot.render(r, 12, r->screen_height() - 92, 280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space)) {
            m_train.store(!m_train.load());
        }
        if (r->is_key_pressed(Rendering::keys::A)) // refit on the live window
            m_want_data = true;
        if (r->is_key_pressed(Rendering::keys::Up)) {
            m_latent = std::min(m_latent + 1, 32);
            request_rebuild();
        }
        if (r->is_key_pressed(Rendering::keys::Down)) {
            m_latent = std::max(m_latent - 1, 1);
            request_rebuild();
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

    static void init_field(Rendering::FieldView &fv, int nx, int ny, bool bar,
                           int ss) {
        fv.init(nx, ny,
                {.supersample = ss,
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

    // ---- worker plumbing ----
    void start_worker() {
        m_stop.store(false);
        m_worker = std::thread([this] { worker_loop(); });
    }

    void stop_worker() {
        m_stop.store(true);
        m_cv.notify_all();
        if (m_worker.joinable())
            m_worker.join();
        // reset shared state so a restart is clean
        std::lock_guard<std::mutex> lk(m_mtx);
        m_train_snaps.resize(0, 0);
        m_live_state.resize(0);
        m_pub_recon.resize(0);
        m_pub_latent.resize(0);
        m_data_dirty = m_rebuild = m_pub_dirty = false;
    }

    void request_rebuild() {
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_rebuild = true;
        }
        m_cv.notify_all();
    }

    // owns the CAE; all encode/decode/train stays on this thread
    void worker_loop() {
        AI::ConvolutionalAutoencoder cae;
        AI::ConvolutionalAutoencoder::TrainConfig cfg{16, 1e-3};
        bool built = false;
        VectorXd mu;         // training-set mean, for recon-error denominator
        MatrixXd last_snaps; // worker-local copy of the frozen window
        uint32_t seed = 1;

        while (!m_stop.load()) {
            MatrixXd snaps;
            VectorXd live;
            bool rebuild = false, have_new_data = false;
            int latent = 0;
            {
                std::unique_lock<std::mutex> lk(m_mtx);
                m_cv.wait_for(lk, std::chrono::milliseconds(5), [this] {
                    return m_stop.load() || m_data_dirty || m_rebuild;
                });
                if (m_stop.load())
                    break;
                rebuild = m_rebuild;
                m_rebuild = false;
                if (m_data_dirty) {
                    snaps = m_train_snaps;
                    m_data_dirty = false;
                    have_new_data = true;
                }
                live = m_live_state;
                latent = m_latent;
            }

            if (rebuild)
                built = false; // force a fresh net at the new latent size

            if (!built && (have_new_data || last_snaps.cols() > 0)) {
                const MatrixXd &data = have_new_data ? snaps : last_snaps;
                if (data.cols() > 0) {
                    cae.build(2, CX, CY, {8, 16, 32}, latent, seed++);
                    cae.set_data(data);
                    mu = data.rowwise().mean();
                    last_snaps = data;
                    built = true;
                }
            } else if (have_new_data && built) {
                last_snaps = snaps; // refit window (resets Adam)
                cae.set_data(snaps);
                mu = snaps.rowwise().mean();
            }

            if (!built || !m_train.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
                continue;
            }

            const double loss = cae.train_epoch(cfg);

            VectorXd recon, latent_vec;
            double err = 0.0;
            if (live.size() == 2 * CX * CY) {
                MatrixXd x = live;           // (state_dim x 1)
                MatrixXd z = cae.encode(x);  // (latent x 1)
                MatrixXd xh = cae.decode(z); // (state_dim x 1)
                recon = xh.col(0);
                latent_vec = z.col(0);
                const double num = (live - recon).squaredNorm();
                const double den = (live - mu).squaredNorm();
                err = den > 1e-12 ? num / den : 0.0;
            }

            {
                std::lock_guard<std::mutex> lk(m_mtx);
                m_pub_recon = std::move(recon);
                m_pub_latent = std::move(latent_vec);
                m_pub_loss = loss;
                m_pub_err = err;
                m_pub_epochs = cae.train_steps();
                m_pub_dirty = true;
            }
        }
    }

    void draw_reconstruction(Rendering::Renderer *r, const Vector2d &o,
                             double vmax) {
        const double rox = o.x();
        const double roy = o.y() + H + FGAP;
        if (!m_have_recon) {
            label(r, rox + W, roy + H, "CAE RECONSTRUCTION  (warming up)",
                  Rendering::palette::text_dim());
            return;
        }
        constexpr int nc = CX * CY;
        m_recon_field.render(r, rox, roy, CCELL,
                             [this, vmax, rox, roy](double wx, double wy,
                                                    double &val, double &a) {
                                 const int i = (int)((wx - rox) / CCELL);
                                 const int j = (int)((wy - roy) / CCELL);
                                 if (i < 0 || i >= CX || j < 0 || j >= CY) {
                                     a = 0.0;
                                     return;
                                 }
                                 const int cc = i + j * CX;
                                 const double u = m_recon[cc],
                                              v = m_recon[nc + cc];
                                 val = std::hypot(u, v) / vmax;
                                 a = freestream_alpha(u, v) *
                                     Rendering::window_alpha(
                                         wx - rox, wy - roy, 0.0, 0.0, W, H,
                                         EDGE_PAD);
                             });
        const Vector2d mc = m_center - o + Vector2d(rox, roy);
        r->draw_circle(mc.x(), mc.y(), RADIUS,
                       Rendering::palette::foreground());
        label(r, rox + W, roy + H / 1.5,
              sfmt("CAE RECONSTRUCTION  %dx%d", CX, CY),
              Rendering::palette::accent2(), 16, 1);
        label(r, rox + W, roy + H / 1.5, sfmt("Latent dim %d", m_latent),
              Rendering::palette::accent2(), 16, 0);
        label(r, rox + W, roy + H / 2,
              sfmt("Recon err %.1f%%", 100.0 * m_recon_err),
              Rendering::palette::accent3(), 16, 0);
    }

    static Rendering::Color node_color(double t) {
        t = std::clamp(t, -1.0, 1.0);
        const Rendering::Color mid = Rendering::palette::background();
        return t >= 0 ? Rendering::color_lerp(mid,
                                              Rendering::palette::accent1(), t)
                      : Rendering::color_lerp(
                            mid, Rendering::palette::accent4(), -t);
    }

    // latent code drawn as a row of value-coloured nodes under the recon
    void draw_latent(Rendering::Renderer *r, const Vector2d &o) {
        if (m_latent_vec.size() == 0)
            return;
        const int n = (int)m_latent_vec.size();
        double m = m_latent_vec.cwiseAbs().maxCoeff();
        if (m < 1e-9)
            m = 1.0;
        const double y = o.y() + H + FGAP - 0.8;
        const double x0 = o.x() + W * 0.5 - (n - 1) * 0.5 * 0.5;
        const Rendering::Color clear{0, 0, 0, 0};
        for (int i = 0; i < n; i++) {
            const double t = m_latent_vec[i] / m;
            const double rr = 0.11 * (0.6 + 0.8 * std::abs(t));
            r->draw_disk(x0 + i * 0.5, y, 0.0, rr, node_color(t), clear);
        }
        label(r, x0 - 0.4, y + 0.45, "latent code",
              Rendering::palette::text_dim(), 14);
    }

    void draw_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("KARMAN CAE", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Snapshots: %d / %d",
                 m_recorder.history().size(), N_CAP);
        if (!m_have_recon && !m_want_data)
            hud.small_text("building net...", Rendering::palette::accent1());
        if (m_have_recon) {
            hud.line(Rendering::palette::text(), "grid %dx%d  latent %d", CX,
                     CY, m_latent);
            hud.line(m_train.load() ? Rendering::palette::accent1()
                                    : Rendering::palette::text_dim(),
                     m_train.load() ? "TRAINING  steps %d" : "paused  steps %d",
                     m_epochs);
            hud.line(Rendering::palette::accent3(), "loss %.1f  recon %.0f%%",
                     m_loss, 100.0 * m_recon_err);
        } else {
            hud.small_text("collecting snapshots...",
                           Rendering::palette::text_dim());
        }
        hud.separator();
        hud.small_text(
            "[F] refit on live flow  [Space] train  [Up/Down] latent",
            Rendering::palette::text_dim());
        hud.small_text("[R] reset", Rendering::palette::text_dim());
    }

    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS, CELL, VISC, 0.0, Vector2d(OX, OY)};

    Vector2d m_center = Vector2d::Zero();
    Rendering::FieldView m_field, m_recon_field;
    Rendering::PlotWidget m_loss_plot;

    AI::SnapshotRecorder m_recorder{CX,     CY,    CCELL,     Vector2d(OX, OY),
                                    STRIDE, N_CAP, TRANSIENT, 2};

    // main-thread render copies
    VectorXd m_recon, m_latent_vec;
    double m_loss = 0.0, m_recon_err = 0.0;
    int m_epochs = 0;
    bool m_have_recon = false;
    bool m_want_data = true;
    int m_latent = 6;
    double m_time = 0.0;
    int m_frame = 0;

    // worker + shared state
    std::thread m_worker;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    std::atomic<bool> m_stop{false};
    std::atomic<bool> m_train{true};

    MatrixXd m_train_snaps; // in: frozen training window
    VectorXd m_live_state;  // in: field to reconstruct
    bool m_data_dirty = false, m_rebuild = false;

    VectorXd m_pub_recon, m_pub_latent; // out: latest reconstruction
    double m_pub_loss = 0.0, m_pub_err = 0.0;
    int m_pub_epochs = 0;
    bool m_pub_dirty = false;
    Solver::BeamBending beam;
};

} // namespace manifold::Demo
