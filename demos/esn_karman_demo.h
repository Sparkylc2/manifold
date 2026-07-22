#pragma once

#include "manifold/ai/autoencoder.h"
#include "manifold/ai/echo_state_network.h"
#include "manifold/ai/snapshot_recorder.h"
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/plot_widget.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

// Live Karman -> coarse autoencoder latent -> ESN forecast. The AE trains on a
// background thread (full field, 2x-coarsened), then freezes; the ESN fits the
// latent dynamics and forecasts them closed-loop. A forecast "episode" freezes
// the flow, rolls the ESN ahead in the reconstruction panel at real time, then
// lets the real flow catch up.
class ESNKarmanDemo : public DemoBase {
  public:
    // full-res fluid + live display
    static constexpr int COLS = 200;
    static constexpr int ROWS = 100;
    static constexpr double CELL = 0.09;
    static constexpr int SS = 2;
    static constexpr double OX = -COLS * CELL * 0.5;
    static constexpr double OY = -ROWS * CELL * 0.5;
    static constexpr double W = COLS * CELL;
    static constexpr double H = ROWS * CELL;

    // coarse grid the AE actually sees (2x per axis)
    static constexpr int CX = COLS / 2;
    static constexpr int CY = ROWS / 2;
    static constexpr double CCELL = W / CX; // same world footprint as the flow

    static constexpr double INFLOW = 3.0;
    static constexpr double VISC = 0.0;
    static constexpr double RADIUS = 0.4;
    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int FADE_PX = 18;

    static constexpr int N_CAP = 160; // AE training window
    static constexpr uint STRIDE = 3;
    static constexpr double TRANSIENT = 4.0;
    static constexpr int MIN_SNAPS = 96;

    static constexpr int RECON_EVERY = 2;
    static constexpr int PUBLISH_EVERY = 15;   // worker epochs between publishes
    static constexpr int AE_FREEZE_STEPS = 3000; // epochs before the AE freezes

    // forecaster
    static constexpr int LATENT_CAP = 2000;
    static constexpr int ESN_MIN = 300; // latents needed before a fit
    static constexpr double AUTO_DWELL = 3.0;

    static constexpr double FGAP = 1.5;

    enum class Phase { Live, Forecast, Catchup };

    const char *name() const override { return "Karman ESN"; }
    double default_cam_x() const override { return 9.75; }
    double default_cam_y() const override { return -4.75; }
    double default_cam_zoom() const override { return 30.0; }

    ~ESNKarmanDemo() override { stop_worker(); }

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
        m_ae_have = false;
        m_frozen = false;
        m_esn_ready = false;
        m_recon.resize(0);
        m_latent_vec.resize(0);
        m_state_mean.resize(0);
        m_last_loss = m_recon_err = 0.0;
        m_epochs = 0;

        m_esn.reset();
        m_latent_hist.clear();
        m_phase = Phase::Live;
        m_auto = false;
        m_forecast_steps = 120;
        m_fc_i = 0;
        m_fc_accum = 0.0;
        m_last_pred.resize(0);
        m_dwell = 0.0;
        m_freeze_time = 0.0;
        m_catchup_target = 0.0;
        m_snap_dt = STRIDE * (1.0 / 60.0);

        init_field(m_field, COLS, ROWS, true, SS);
        init_field(m_recon_field, CX, CY, false, SS);
        m_loss_plot.configure("AE loss", Rendering::palette::accent3(), 800);
        m_loss_plot.clear();

        start_worker();
    }

    void process(double dt) override {
        m_snap_dt = STRIDE * dt;

        if (m_phase == Phase::Forecast) {
            forecast_tick(dt);
            return; // flow frozen while the ESN rolls ahead
        }

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

        // pull the latest AE the worker published
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            if (m_pub_dirty) {
                m_ae = m_pub_ae;
                m_state_mean = m_pub_mu;
                m_last_loss = m_pub_loss;
                m_epochs = m_pub_epochs;
                m_ae_have = true;
                m_loss_plot.push(m_last_loss);
                m_pub_dirty = false;
            }
        }

        // freeze the AE once it settles, so the ESN sees a stable latent space
        if (m_ae_have && !m_frozen && m_epochs >= AE_FREEZE_STEPS)
            freeze_ae();

        // collect the frozen latent series for the ESN, at snapshot cadence
        if (m_frozen && m_frame % STRIDE == 0)
            push_latent();

        if (m_frozen && !m_esn_ready && (int)m_latent_hist.size() >= ESN_MIN)
            build_esn();

        if (m_phase == Phase::Live && (m_ae_have || m_frozen) &&
            m_frame % RECON_EVERY == 0)
            update_reconstruction();

        if (m_phase == Phase::Catchup) {
            if (m_time >= m_catchup_target) {
                m_phase = Phase::Live;
                m_dwell = 0.0;
            }
        } else { // Live
            m_dwell += dt;
            if (m_auto && m_esn_ready && m_dwell >= AUTO_DWELL)
                start_forecast();
        }
    }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW;

        m_field.render(
            r, o.x(), o.y(), CELL,
            [this, vmax](double wx, double wy, double &val, double &a) {
                Vector2d v;
                m_fluid.velocity_at(Vector2d(wx, wy), &v,
                                    Fluid::Interp::Linear);
                val = v.norm() / vmax;
                a = freestream_alpha(v.x(), v.y());
            });
        r->draw_circle(m_center.x(), m_center.y(), RADIUS,
                       Rendering::palette::foreground());
        const bool frozen_view = m_phase != Phase::Live;
        label(r, o.x() + W, o.y() + H / 1.5,
              frozen_view ? "LIVE FLOW (frozen)" : "LIVE FLOW",
              frozen_view ? Rendering::palette::text_dim()
                          : Rendering::palette::accent2());

        draw_reconstruction(r, o, vmax);
        draw_latent(r, o);
        draw_hud(r);
        m_loss_plot.render(r, 12, r->screen_height() - 92, 280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::F))
            refit_ae();
        if (r->is_key_pressed(Rendering::keys::E))
            refit_esn();
        if (r->is_key_pressed(Rendering::keys::Space) && m_esn_ready)
            start_forecast();
        if (r->is_key_pressed(Rendering::keys::A))
            m_auto = !m_auto;
        if (r->is_key_pressed(Rendering::keys::Up))
            m_forecast_steps = std::min(m_forecast_steps + 10, LATENT_CAP);
        if (r->is_key_pressed(Rendering::keys::Down))
            m_forecast_steps = std::max(m_forecast_steps - 10, 1);
        // live ESN tuning (rebuilds the reservoir on the current latents)
        if (r->is_key_pressed(Rendering::keys::Left)) {
            m_esn_sigma = std::max(m_esn_sigma * 0.7, 1e-4);
            retune_esn();
        }
        if (r->is_key_pressed(Rendering::keys::Right)) {
            m_esn_sigma = std::min(m_esn_sigma / 0.7, 10.0);
            retune_esn();
        }
        if (r->is_key_pressed(Rendering::keys::Z)) {
            m_esn_rho = std::max(m_esn_rho - 0.02, 0.05);
            retune_esn();
        }
        if (r->is_key_pressed(Rendering::keys::X)) {
            m_esn_rho = std::min(m_esn_rho + 0.02, 0.999);
            retune_esn();
        }
    }

  private:
    // ---- forecaster ------------------------------------------------------
    void freeze_ae() {
        m_ae_fc = m_ae; // fix the encoder the ESN will use
        m_frozen = true;
        m_train.store(false); // pause background training
        m_latent_hist.clear();
        m_esn_ready = false;
    }

    void refit_ae() { // retrain the AE on a fresh live window
        m_want_data = true;
        m_train.store(true);
        m_frozen = false;
        m_esn_ready = false;
        m_latent_hist.clear();
        m_phase = Phase::Live;
    }

    void refit_esn() { // recollect latents and refit (freeze AE first if needed)
        if (!m_ae_have)
            return;
        if (!m_frozen)
            freeze_ae();
        else {
            m_esn_ready = false;
            m_latent_hist.clear();
        }
    }

    void retune_esn() { // rebuild reservoir with current knobs, same latents
        if (m_frozen && (int)m_latent_hist.size() >= ESN_MIN)
            build_esn();
    }

    void push_latent() {
        const VectorXd x = m_recorder.sample_state(m_fluid); // coarse state
        m_latent_hist.push_back(m_ae_fc.encode(x));
        if ((int)m_latent_hist.size() > LATENT_CAP)
            m_latent_hist.pop_front();
    }

    void build_esn() {
        const int n = (int)m_latent_hist.size();
        if (n < ESN_MIN || !m_frozen)
            return;

        MatrixXd Z(n, m_latent); // latent trajectory, time on rows
        for (int t = 0; t < n; t++)
            Z.row(t) = m_latent_hist[t].transpose();

        AI::ESN::ESNConfig cfg;
        cfg.N_r = m_esn_units;
        cfg.N_wash = m_esn_wash;
        cfg.N_split = 4;
        cfg.upsample = 1;
        cfg.norm_method = AI::ESN::Range;
        cfg.W_in_type = AI::ESN::Sparse;
        cfg.observed_idx.resize(m_latent);
        std::iota(cfg.observed_idx.begin(), cfg.observed_idx.end(), 0);
        cfg.add_noise = true;
        cfg.noise = m_esn_noise;
        cfg.rho = m_esn_rho;
        cfg.sigma_in = m_esn_sigma;
        cfg.tikh = 1e-6;
        cfg.seed = m_seed++;

        // size the wtv/test split to the series (upsample = 1 -> dt_ESN =
        // m_snap_dt, so N_train = round(t_train / dt_ESN) = n_train)
        const int n_train = std::max(1, (int)std::llround(0.70 * n));
        const int n_val = std::max(1, (int)std::llround(0.15 * n));
        cfg.t_train = n_train * m_snap_dt;
        cfg.t_val = n_val * m_snap_dt;

        m_esn.emplace(m_snap_dt, MatrixXd::Zero(m_latent, 1), cfg);
        m_esn->train({Z});
        m_esn_ready = true;
    }

    void start_forecast() {
        if (m_phase != Phase::Live || !m_esn_ready)
            return;
        m_esn->reset_state();
        const int n = (int)m_latent_hist.size();
        const int w = std::min(m_esn_wash, n);
        for (int t = n - w; t < n; t++)
            m_esn->advance(m_latent_hist[t]);
        m_freeze_time = m_time;
        m_fc_i = 0;
        m_fc_accum = 0.0;
        m_last_pred.resize(0);
        m_phase = Phase::Forecast;
    }

    void forecast_tick(double dt) {
        if (!m_esn_ready) {
            m_phase = Phase::Live;
            return;
        }
        // advance at real time: one ESN step (= dt_ESN of latent time) per
        // dt_ESN of wall time, so the rollout evolves no faster than the flow
        m_fc_accum += dt;
        while (m_fc_accum >= m_snap_dt && m_fc_i < m_forecast_steps) {
            m_last_pred = m_esn->predict_step(); // (latent x 1)
            m_fc_accum -= m_snap_dt;
            m_fc_i++;
        }
        if (m_last_pred.size())
            m_recon = m_ae_fc.decode(m_last_pred);
        if (m_fc_i >= m_forecast_steps) {
            m_phase = Phase::Catchup;
            m_catchup_target = m_freeze_time + m_forecast_steps * m_snap_dt;
        }
    }

    void update_reconstruction() {
        const AI::Autoencoder &ae = m_frozen ? m_ae_fc : m_ae;
        const VectorXd x = m_recorder.sample_state(m_fluid);
        const VectorXd z = ae.encode(x);
        m_recon = ae.decode(z);
        m_latent_vec = z;
        const double num = (x - m_recon).squaredNorm();
        const double den = m_state_mean.size() == x.size()
                               ? (x - m_state_mean).squaredNorm()
                               : 0.0;
        m_recon_err = den > 1e-12 ? num / den : 0.0;
    }

    // ---- background AE worker (owns the training net) --------------------
    void start_worker() {
        m_stop.store(false);
        m_train.store(true);
        m_worker = std::thread([this] { worker_loop(); });
    }

    void stop_worker() {
        m_stop.store(true);
        m_cv.notify_all();
        if (m_worker.joinable())
            m_worker.join();
        std::lock_guard<std::mutex> lk(m_mtx);
        m_train_snaps.resize(0, 0);
        m_data_dirty = m_rebuild = m_pub_dirty = false;
    }

    void publish(const AI::Autoencoder &ae, const VectorXd &mu, double loss,
                 int ep) {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_pub_ae = ae;
        m_pub_mu = mu;
        m_pub_loss = loss;
        m_pub_epochs = ep;
        m_pub_dirty = true;
    }

    void worker_loop() {
        AI::Autoencoder ae;
        AI::Autoencoder::TrainConfig cfg{32, 1e-2};
        bool built = false;
        VectorXd mu;
        MatrixXd last_snaps;
        uint32_t seed = 1;
        int since_pub = 0;

        while (!m_stop.load()) {
            MatrixXd snaps;
            bool have_new = false;
            int latent = 0;
            {
                std::unique_lock<std::mutex> lk(m_mtx);
                m_cv.wait_for(lk, std::chrono::milliseconds(5), [this] {
                    return m_stop.load() || m_data_dirty;
                });
                if (m_stop.load())
                    break;
                latent = m_latent;
                if (m_data_dirty) {
                    snaps = m_train_snaps;
                    m_data_dirty = false;
                    have_new = true;
                }
            }

            if (have_new) {
                if (!built) {
                    ae.build(2 * CX * CY, {16, 8}, latent, seed++);
                    built = true;
                }
                ae.set_data(snaps); // (re)fit window, resets Adam
                mu = snaps.rowwise().mean();
                last_snaps = snaps;
                since_pub = 0;
                publish(ae, mu, 0.0, 0); // give main an AE immediately
            }

            if (!built || !m_train.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
                continue;
            }

            const double loss = ae.train_epoch(cfg);
            if (++since_pub >= PUBLISH_EVERY) {
                since_pub = 0;
                publish(ae, mu, loss, ae.train_steps());
            }
        }
    }

    // ---- helpers ---------------------------------------------------------
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

    void draw_reconstruction(Rendering::Renderer *r, const Vector2d &o,
                             double vmax) {
        const double rox = o.x();
        const double roy = o.y() + H + FGAP;
        if (m_recon.size() == 0) {
            label(r, rox + W, roy + H, "RECONSTRUCTION  (warming up)",
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
                                 a = freestream_alpha(u, v);
                             });
        const Vector2d mc = m_center - o + Vector2d(rox, roy);
        r->draw_circle(mc.x(), mc.y(), RADIUS,
                       Rendering::palette::foreground());

        if (m_phase == Phase::Forecast) {
            label(r, rox + W, roy + H / 1.5,
                  sfmt("ESN FORECAST  +%.1fs", m_fc_i * m_snap_dt),
                  Rendering::palette::accent1(), 16, 1);
            label(r, rox + W, roy + H / 2,
                  sfmt("step %d / %d", m_fc_i, m_forecast_steps),
                  Rendering::palette::accent1(), 16, 0);
        } else if (m_phase == Phase::Catchup) {
            label(r, rox + W, roy + H / 1.5, "ESN FORECAST (held)",
                  Rendering::palette::accent1(), 16, 1);
            label(r, rox + W, roy + H / 2, "reality catching up...",
                  Rendering::palette::text_dim(), 16, 0);
        } else {
            label(r, rox + W, roy + H / 1.5, "AE RECONSTRUCTION",
                  Rendering::palette::accent2(), 16, 1);
            label(r, rox + W, roy + H / 2,
                  sfmt("Recon err %.1f%%", 100.0 * m_recon_err),
                  Rendering::palette::accent3(), 16, 0);
        }
    }

    static Rendering::Color node_color(double t) {
        t = std::clamp(t, -1.0, 1.0);
        const Rendering::Color mid = Rendering::palette::background();
        return t >= 0 ? Rendering::color_lerp(mid,
                                              Rendering::palette::accent1(), t)
                      : Rendering::color_lerp(
                            mid, Rendering::palette::accent4(), -t);
    }

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
        hud.title("KARMAN ESN", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Snapshots: %d / %d",
                 m_recorder.history().size(), N_CAP);

        if (!m_ae_have) {
            hud.small_text("training autoencoder (bg)...",
                           Rendering::palette::accent1());
        } else if (!m_frozen) {
            hud.line(Rendering::palette::accent1(), "AE training  epochs %d",
                     m_epochs);
            hud.line(Rendering::palette::accent3(), "loss %.4f  recon %.0f%%",
                     m_last_loss, 100.0 * m_recon_err);
        } else if (!m_esn_ready) {
            hud.line(Rendering::palette::text(), "AE frozen  latents %d / %d",
                     (int)m_latent_hist.size(), ESN_MIN);
            hud.small_text("collecting latent series...",
                           Rendering::palette::accent1());
        } else {
            const char *ph = m_phase == Phase::Forecast   ? "FORECASTING"
                             : m_phase == Phase::Catchup   ? "CATCH-UP"
                                                           : "LIVE";
            hud.line(Rendering::palette::accent1(), "Forecaster ready  [%s]",
                     ph);
            hud.line(Rendering::palette::text(), "Horizon %d  ~%.1fs  |  N_r %d",
                     m_forecast_steps, m_forecast_steps * m_snap_dt,
                     m_esn_units);
            hud.line(Rendering::palette::text_dim(),
                     "sigma_in %.3g  rho %.2f", m_esn_sigma, m_esn_rho);
            hud.line(m_auto ? Rendering::palette::accent1()
                            : Rendering::palette::text_dim(),
                     m_auto ? "AUTO on" : "auto off");
        }
        hud.separator();
        hud.small_text("[Space] forecast  [A] auto  [Up/Down] horizon",
                       Rendering::palette::text_dim());
        hud.small_text("[Left/Right] sigma  [Z/X] rho  [E] refit ESN",
                       Rendering::palette::text_dim());
        hud.small_text("[F] retrain AE  [R] reset",
                       Rendering::palette::text_dim());
    }

    // ---- state -----------------------------------------------------------
    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS, CELL, VISC, 0.0, Vector2d(OX, OY)};

    Vector2d m_center = Vector2d::Zero();
    Rendering::FieldView m_field, m_recon_field;
    Rendering::PlotWidget m_loss_plot;

    AI::SnapshotRecorder m_recorder{CX,     CY,    CCELL,     Vector2d(OX, OY),
                                    STRIDE, N_CAP, TRANSIENT, 2};

    // main-thread AE copies
    AI::Autoencoder m_ae;    // latest published (for display before freeze)
    AI::Autoencoder m_ae_fc; // frozen encoder the ESN uses
    VectorXd m_recon, m_latent_vec, m_state_mean;
    double m_last_loss = 0.0, m_recon_err = 0.0;
    int m_epochs = 0;
    bool m_ae_have = false, m_frozen = false;
    bool m_want_data = true;
    int m_latent = 6;
    uint32_t m_seed = 1;
    double m_time = 0.0;
    int m_frame = 0;

    // forecaster
    std::optional<AI::ESN> m_esn;
    bool m_esn_ready = false;
    std::deque<VectorXd> m_latent_hist;
    Phase m_phase = Phase::Live;
    bool m_auto = false;
    int m_forecast_steps = 120;
    int m_fc_i = 0;
    double m_fc_accum = 0.0;
    VectorXd m_last_pred;
    double m_dwell = 0.0;
    double m_freeze_time = 0.0;
    double m_catchup_target = 0.0;
    double m_snap_dt = STRIDE * (1.0 / 60.0);

    // ESN tunables (live-adjustable)
    int m_esn_units = 300;
    int m_esn_wash = 40;
    double m_esn_sigma = 0.1;
    double m_esn_rho = 0.9;
    double m_esn_noise = 1e-3;

    // worker + shared state
    std::thread m_worker;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    std::atomic<bool> m_stop{false};
    std::atomic<bool> m_train{true};

    MatrixXd m_train_snaps; // in: frozen training window
    bool m_data_dirty = false, m_rebuild = false;

    AI::Autoencoder m_pub_ae; // out: latest trained AE
    VectorXd m_pub_mu;
    double m_pub_loss = 0.0;
    int m_pub_epochs = 0;
    bool m_pub_dirty = false;
};

} // namespace manifold::Demo
