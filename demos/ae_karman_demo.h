#pragma once

#include "manifold/ai/autoencoder.h"
#include "manifold/ai/pod.h"
#include "manifold/ai/snapshot_recorder.h"
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/plot_widget.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <future>
#include <string>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

class AEKarmanDemo : public DemoBase {
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
    static constexpr int FADE_PX = 18;
    static constexpr double EDGE_PAD = 1.2;

    static constexpr int N_CAP = 120;
    static constexpr uint STRIDE = 3;
    static constexpr double TRANSIENT = 4.0;
    static constexpr int MIN_SNAPS = 100;

    static constexpr int POD_D = 8;

    static constexpr int EPOCHS_PER_FRAME = 20;
    static constexpr int RECON_EVERY = 3;

    // layout
    static constexpr double FGAP = 1.5, NGAP = 1.5;
    static constexpr double NW = 16.0, NH = 8.0;
    static constexpr double NODE_DY = 0.55, NODE_R = 0.11;

    const char *name() const override { return "Karman AE"; }
    double default_cam_x() const override { return 9.75; }
    double default_cam_y() const override { return -4.75; }
    double default_cam_zoom() const override { return 30.0; }

    void initialize() override {
        if (m_pod_job.valid())
            m_pod_job.wait();
        m_pod_running = false;

        m_fluid.clear();
        m_fluid.set_channel(INFLOW);
        const Vector2d o = m_fluid.origin();
        m_center = o + Vector2d(0.30 * W, 0.5 * H + 0.04);
        m_fluid.set_circle_obstacle(m_center, RADIUS);

        m_recorder =
            AI::SnapshotRecorder(COLS, ROWS, CELL, o, STRIDE, N_CAP, TRANSIENT);
        m_pod_ready = m_ae_built = m_training = false;
        m_want_refit = true;
        m_D = 0;
        m_last_loss = m_recon_err = 0.0;
        m_time = 0.0;
        m_frame = 0;
        m_recon.resize(0);
        m_acts.clear();

        init_field(m_field, true, SS);
        init_field(m_recon_field, false, 1);
        m_loss_plot.configure("AE loss", Rendering::palette::accent3(), 800);
        m_loss_plot.clear();
    }

    void process(double dt) override {
        m_fluid.advance(dt);
        m_time += dt;
        m_recorder.maybe_capture(m_fluid, m_time);
        m_frame++;

        if (m_want_refit && !m_pod_running &&
            m_recorder.history().size() >= MIN_SNAPS) {
            MatrixXd snap = m_recorder.history().matrix();
            m_pod_job =
                std::async(std::launch::async, [snap = std::move(snap)]() {
                    AI::POD p;
                    p.compute(snap);
                    return p;
                });
            m_pod_running = true;
        }
        if (m_pod_running && m_pod_job.wait_for(std::chrono::seconds(0)) ==
                                 std::future_status::ready) {
            m_pod = m_pod_job.get();
            m_D = std::min(POD_D, m_pod.num_modes());
            m_pod.set_rank(m_D);
            m_C = m_pod.coeff_matrix().topRows(m_D);
            build_ae();
            m_pod_ready = true;
            m_pod_running = false;
            m_want_refit = false;
        }

        if (m_training && m_ae_built)
            for (int k = 0; k < EPOCHS_PER_FRAME; k++) {
                m_last_loss = m_ae.train_epoch(m_cfg);
                m_loss_plot.push(m_last_loss);
            }

        if (m_pod_ready && m_ae_built && m_frame % RECON_EVERY == 0)
            update_reconstruction();
    }

    void render(Rendering::Renderer *r) override {
        // draw_world_grid(r);
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
        draw_network(r, o);
        draw_hud(r);
        m_loss_plot.render(r, 12, r->screen_height() - 92, 280, 80);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space) && m_ae_built)
            m_training = !m_training;
        if (r->is_key_pressed(Rendering::keys::N) && m_pod_ready)
            build_ae();
        if (r->is_key_pressed(Rendering::keys::F))
            m_want_refit = true;
        if (r->is_key_pressed(Rendering::keys::Up) && m_pod_ready) {
            m_latent++;
            build_ae();
        }
        if (r->is_key_pressed(Rendering::keys::Down) && m_pod_ready) {
            m_latent--;
            build_ae();
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

    void build_ae() {
        m_latent = std::clamp(m_latent, 1, m_D);
        m_ae = AI::Autoencoder();
        m_ae.build(m_D, {12, 8}, m_latent, /*seed*/ m_seed++);
        m_ae.set_data(m_C);
        m_ae_built = true;
        m_training = true;
        m_loss_plot.clear();
    }

    void update_reconstruction() {
        const VectorXd x = m_recorder.sample_state(m_fluid);
        const VectorXd c = m_pod.encode(x); // D POD coeffs
        const VectorXd z = m_ae.encode(c);  // latent
        const VectorXd ch = m_ae.decode(z); // D coeffs
        m_recon = m_pod.decode(ch);         // back to a field
        m_acts = m_ae.activations(c);       // for the network diagram

        const double num = (x - m_recon).squaredNorm();
        const double den = (x - m_pod.mean()).squaredNorm();
        m_recon_err = den > 1e-12 ? num / den : 0.0;
    }

    void draw_reconstruction(Rendering::Renderer *r, const Vector2d &o,
                             double vmax) {
        const double rox = o.x();
        const double roy = o.y() + H + FGAP;
        if (!m_pod_ready || m_recon.size() == 0) {
            label(r, rox + W, roy + H, "RECONSTRUCTION  (warming up)",
                  Rendering::palette::text_dim());
            return;
        }
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
                                 const int cc = i + j * COLS;
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
        label(r, rox + W, roy + H / 1.5, "AE RECONSTRUCTION",
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

    void draw_network(Rendering::Renderer *r, const Vector2d &o) {
        if (m_acts.empty())
            return;

        const int L = (int)m_acts.size();
        // const double cx = o.x() + W + FGAP * 0.5;
        const double cx = o.x() + W / 1.75;
        const double x0 = cx - NW * 0.5;
        const double cy = (o.y() - NGAP) - NH * 0.5;
        const double dx = NW / (L - 1);

        auto node_y = [&](int i, int n) {
            return cy + ((n - 1) * 0.5 - i) * NODE_DY;
        };

        const Rendering::Color ec = Rendering::palette::grid_line();
        for (int l = 0; l < L - 1; l++) {
            const int n1 = (int)m_acts[l].size(),
                      n2 = (int)m_acts[l + 1].size();
            const double xa = x0 + l * dx, xb = x0 + (l + 1) * dx;
            for (int i = 0; i < n1; i++)
                for (int j = 0; j < n2; j++)
                    r->draw_line(xa, node_y(i, n1), xb, node_y(j, n2), 0.6f,
                                 ec);
        }

        const Rendering::Color clear{0, 0, 0, 0};
        for (int l = 0; l < L; l++) {
            const VectorXd &a = m_acts[l];
            const int n = (int)a.size();
            double m = a.cwiseAbs().maxCoeff();
            if (m < 1e-9)
                m = 1.0;
            const double x = x0 + l * dx;
            for (int i = 0; i < n; i++) {
                const double t = a[i] / m;
                const double rr = NODE_R * (0.6 + 0.8 * std::abs(t));
                r->draw_disk(x, node_y(i, n), 0.0, rr, node_color(t), clear);
            }
        }
        label(r, x0 + W * 0.86, cy + NH * 0.5 + 0.4, "AUTOENCODER",
              Rendering::palette::accent2());
        label(r, x0 + (L / 2) * dx - 0.5, cy - NH * 0.5 - 0.2, "latent",
              Rendering::palette::text_dim(), 14);
    }

    void draw_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("KARMAN AE", Rendering::palette::accent2());
        hud.line(Rendering::palette::text(), "Snapshots: %d / %d",
                 m_recorder.history().size(), N_CAP);
        if (m_pod_running)
            hud.small_text("computing POD...", Rendering::palette::accent1());
        if (m_ae_built) {
            hud.line(Rendering::palette::text(), "POD dims: %d  latent: %d",
                     m_D, m_latent);
            hud.line(m_training ? Rendering::palette::accent1()
                                : Rendering::palette::text_dim(),
                     m_training ? "TRAINING  epochs %d" : "paused  epochs %d",
                     m_ae.train_steps());
            hud.line(Rendering::palette::accent3(), "loss %.4f  recon %.0f%%",
                     m_last_loss, 100.0 * m_recon_err);
        } else {
            hud.small_text("collecting snapshots...",
                           Rendering::palette::text_dim());
        }
        hud.separator();
        hud.small_text(
            "[F] refit on live flow  [Space] train  [Up/Down] latent",
            Rendering::palette::text_dim());
        hud.small_text("[N] new net  [R] reset",
                       Rendering::palette::text_dim());
    }

    Fluid::StableFluidSolver m_fluid{
        (unsigned)ROWS, (unsigned)COLS, CELL, VISC, 0.0, Vector2d(OX, OY)};

    Vector2d m_center = Vector2d::Zero();
    Rendering::FieldView m_field, m_recon_field;
    Rendering::PlotWidget m_loss_plot;

    AI::SnapshotRecorder m_recorder{COLS,   ROWS,  CELL,     Vector2d(OX, OY),
                                    STRIDE, N_CAP, TRANSIENT};
    AI::POD m_pod;
    AI::Autoencoder m_ae;
    AI::Autoencoder::TrainConfig m_cfg{32, 1e-2};
    MatrixXd m_C;
    VectorXd m_recon;
    std::vector<VectorXd> m_acts;

    std::future<AI::POD> m_pod_job;
    bool m_pod_running = false, m_pod_ready = false;
    bool m_ae_built = false, m_training = false;
    bool m_want_refit = true;
    int m_D = 0;
    int m_latent = 5;
    uint32_t m_seed = 1;
    double m_last_loss = 0.0, m_recon_err = 0.0;
    double m_time = 0.0;
    int m_frame = 0;
};

} // namespace manifold::Demo
