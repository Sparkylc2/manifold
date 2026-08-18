#pragma once

#include "forecast_demo_spec.h"

#include <manifold/ai/snapshot_recorder.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace manifold::Demo {

namespace FSpec = ForecastSpec;

using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

// One live Karman street, one fixed POD basis, two forecasters
// rolling it forward side by side. Everything is pretrained offline by
// tools/train_forecast_demo.cpp, so nothing here fits anything: the demo
// encodes the live field, drives both models open-loop, and periodically cuts
// them loose for a couple of seconds while the flow holds still. When the
// rollout ends both panels freeze on their last guess and the flow retraces the
// window they ran ahead of, and the truth it uncovers draws the error curve
// underneath -- one point per snapshot, left to right.
class ForecastKarmanDemo : public DemoBase {
  public:
    static constexpr int COLS = FSpec::COLS, ROWS = FSpec::ROWS;
    static constexpr double CELL = FSpec::CELL;
    static constexpr double W = FSpec::W, H = FSpec::H;
    static constexpr int GX = FSpec::GX, GY = FSpec::GY;
    static constexpr double GCELL = FSpec::GCELL;
    static constexpr double INFLOW = FSpec::INFLOW, RADIUS = FSpec::RADIUS;
    static constexpr double SNAP_DT = FSpec::SNAP_DT;
    static constexpr unsigned STRIDE = FSpec::STRIDE;
    static constexpr int RANK = FSpec::POD_RANK;

    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int SS = 2;
    // world-space taper on the visible window. the crop below is what would
    // otherwise put a hard vertical line down the left of every panel, and the
    // grid's own texel fade cannot see it -- the texture runs past the crop
    static constexpr double EDGE_PAD = 2.0;
    static constexpr double FIELD_GAMMA = 0.29;

    // ---- layout, world units ----
    // the inflow sixth of the domain is uniform freestream and says nothing, so
    // every panel is cropped to the same window before it is placed. the model
    // grid and the flow grid cover the identical footprint, so one crop serves
    // all three
    static constexpr double CLIP_L = 0.15, CLIP_R = 1.00;
    static constexpr double PW = W * (CLIP_R - CLIP_L);
    static constexpr double PGAP = 1.2; // esn | lstm
    static constexpr double ROW_W = 2.0 * PW + PGAP;
    // each field carries about a fifth of its height as transparent freestream
    // margin, so the gaps read tighter than they measure — but every panel also
    // hangs two lines of label above itself, and those need the room
    static constexpr double VGAP = 2.2; // live -> pair
    // Labels anchor here rather than at the domain top: the upper part of every
    // panel is uniform freestream, which freestream_alpha() draws at zero and
    // the edge taper softens further, so a caption hung at y = H sits a good
    // world unit clear of anything visible before the 22 px offset even starts.
    static constexpr double LABEL_TOP = H - 1.1;
    // Gutters either side of the plot: the axis reads out of the left one, each
    // trace's final error out of the right, so neither sits on top of a line.
    // Equal on both sides so the BOX is centred on the cell -- and therefore
    // under the pair of panels above it, which together span the full ROW_W.
    // Asymmetric gutters sized to their own text left it visibly off-centre.
    static constexpr double PLOT_GUTTER = 3.5;
    static constexpr double PLOT_X0 = PLOT_GUTTER;
    static constexpr double PLOT_W = ROW_W - 2.0 * PLOT_GUTTER;
    static constexpr double PLOT_GAP = 1.9, PLOT_H = 3.8;
    static constexpr double LIVE_X = 0.0; // left aligned; the right is for text
    static constexpr double PAIR_Y = -(VGAP + H);
    static constexpr double PAIR_LABEL_Y = PAIR_Y + LABEL_TOP;
    static constexpr double PLOT_Y = PAIR_Y - PLOT_GAP - PLOT_H;
    static constexpr double MG_BORDER = 0.012, MG_ROUND = 0.10;

    // ---- cycle ----
    static constexpr int HORIZON = 50;   // rollout steps, 2.5 s at SNAP_DT
    static constexpr double DWELL = 2.0; // live seconds between rollouts
    // sim time of the first rollout. the street has to be as developed as the
    // one the checkpoint was fitted to, and the cell's warmup is set a second
    // short of this so the reel opens on live flow and cuts loose right after
    static constexpr double FIRST_FC = 20.0;
    // sim time after which snapshots count toward the error reference, i.e.
    // once the street is shedding and its fluctuation size has settled
    static constexpr double SIGMA_AFTER = 8.0;

    // published extent, for the same reason the AE cell publishes one: the
    // captions overflow in pixels, and the bounds are what the slot fits
    static constexpr double BOUND_X0 = 0.0, BOUND_X1 = ROW_W;
    static constexpr double BOUND_Y1 = LABEL_TOP + 1.1;
    static constexpr double BOUND_Y0 = PLOT_Y - 0.3;
    static constexpr double ERR_FLOOR = 0.03; // smallest plot ceiling

    enum class Phase { Live, Forecast, Catchup };

    const char *name() const override { return "Karman ESN vs LSTM"; }
    double default_cam_x() const override { return 0.5 * ROW_W; }
    double default_cam_y() const override { return -3.3; }
    double default_cam_zoom() const override { return 24.0; }

    void initialize() override {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);
        m_center = FSpec::center();
        m_fluid.set_circle_obstacle(m_center, RADIUS);
        m_recorder = AI::SnapshotRecorder(GX, GY, GCELL, m_fluid.origin(),
                                          STRIDE, 1, 0.0, 2);

        m_time = 0.0;
        m_driven = 0;
        m_snap_accum = 0.0;
        m_phase = Phase::Live;
        m_next_fc = FIRST_FC;
        m_fc_i = m_catch_i = 0;
        m_fc_accum = 0.0;
        m_pred_esn.clear();
        m_pred_lstm.clear();
        m_curve.clear();
        m_smooth.clear();
        m_out[0].resize(0);
        m_out[1].resize(0);
        m_prev[0].resize(0);
        m_prev[1].resize(0);
        m_live_err[0] = m_live_err[1] = 0.0;
        m_sig_sum = 0.0;
        m_sig_n = 0;

        load_models();

        init_field(m_field, COLS, ROWS);
        init_field(m_panel[0], GX, GY);
        init_field(m_panel[1], GX, GY);
    }

    void process(double dt) override {
        if (m_phase == Phase::Forecast) {
            forecast_tick(dt); // flow held while the models run on their own
            return;
        }

        m_fluid.advance(dt);
        m_time += dt;

        // Snapshot on SIM TIME, never on a frame count. Both models were fitted
        // at one step per SNAP_DT of simulated time, and the launcher hands out
        // a wall-clock dt at 240 fps unless --dt pins it -- so "every third
        // frame" drove them four times too fast, racing the predicted phase
        // ahead of the flow and then beating back as it wrapped.
        // the epsilon matters at the intended --dt 1/60: three of those added
        // up is a rounding step away from SNAP_DT computed as 3*(1/60), and
        // without slack the snapshot would occasionally slip a frame and then
        // catch up
        m_snap_accum += dt;
        while (m_snap_accum >= SNAP_DT - 1e-9) {
            m_snap_accum -= SNAP_DT;
            snapshot();
        }

        if (m_phase == Phase::Live && ready() && m_time >= m_next_fc)
            start_forecast();
    }

    void render(Rendering::Renderer *r) override {
        render_cell(r);
        draw_hud(r);
    }

    void render_cell(Rendering::Renderer *r) override {
        const double vmax = 2.0 * INFLOW;
        const Vector2d o = m_fluid.origin();
        const bool held = m_phase != Phase::Live;

        draw_panel(r, m_field, LIVE_X, 0.0, COLS, ROWS, CELL,
                   [&](int i, int j, double &val, double &a) {
                       Vector2d v;
                       m_fluid.velocity_at(Vector2d(o.x() + (i + 0.5) * CELL,
                                                    o.y() + (j + 0.5) * CELL),
                                           &v, Fluid::Interp::Cubic);
                       val = v.norm() / vmax;
                       a = freestream_alpha(v.x(), v.y());
                   });
        obstacle(r, LIVE_X, 0.0);
        label(r, LIVE_X, LABEL_TOP, held ? "LIVE FLOW  (held)" : "LIVE FLOW",
              held ? Rendering::palette::text_dim()
                   : Rendering::palette::accent2());

        draw_model(r, 0, 0.0, "ECHO STATE NETWORK", vmax);
        draw_model(r, 1, PW + PGAP, "LSTM", vmax);
        draw_plot(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Space) && ready())
            start_forecast();
        if (r->is_key_pressed(Rendering::keys::Up))
            m_horizon = std::min(m_horizon + 10, 400);
        if (r->is_key_pressed(Rendering::keys::Down))
            m_horizon = std::max(m_horizon - 10, 10);
    }

  private:
    struct Point {
        double err[2]; // esn, lstm
    };

    // ---- models ----------------------------------------------------------
    void load_models() {
        // t_train/t_val in the config only size a fit, so the frame count here
        // is irrelevant to a reservoir that is never fitted, only stepped
        m_esn.emplace(SNAP_DT, MatrixXd::Zero(RANK, 1),
                      FSpec::esn_cfg(RANK, 800, 1));
        m_lstm.emplace(RANK, FSpec::LSTM_UNITS, AI::LSTM::LSTMCfg{}, 1);

        m_path = FSpec::weights_path();
        m_loaded = FSpec::load_all(m_path, m_pod, *m_esn, *m_lstm);
        if (!m_loaded) {
            std::printf(
                "[forecast] no checkpoint at %s -- run train_forecast_demo\n",
                m_path.c_str());
            return;
        }
        m_esn->reset_state();
        m_roll = m_lstm->fresh_rollout();
    }

    bool ready() const { return m_loaded && m_driven >= FSpec::LSTM_WASH; }

    double sigma() const {
        return m_sig_n > 0 ? std::sqrt(m_sig_sum / m_sig_n) : 1.0;
    }

    // one snapshot of live truth: encode it and drive both models open-loop
    // with it. during catch-up that is also what pulls them back onto the real
    // trajectory, one step at a time, while the flow retraces the window they
    // just ran ahead of
    void snapshot() {
        if (!m_loaded)
            return;

        const VectorXd x = m_recorder.sample_state(m_fluid);
        const VectorXd z = m_pod.encode(x);

        const VectorXd ze = m_esn->advance(z).col(0);
        const VectorXd zl = m_lstm->advance(m_roll, z);
        m_driven++;

        // reference for the tracking readout: the flow's own RMS fluctuation,
        // accumulated once and held rather than taken instantaneously, which
        // would put the shedding cycle itself in the denominator
        if (m_time > SIGMA_AFTER) {
            m_sig_sum += (x - m_pod.mean()).squaredNorm();
            m_sig_n++;
        }

        // Each model emitted its guess for THIS instant one step ago, so the
        // honest open-loop score compares against what was staged then, not
        // against what it has just produced for the next step. Restaged in
        // every phase, including catch-up: the models are driven there too, and
        // leaving the freeze's guess sitting in m_prev would score the first
        // live snapshot after a catch-up against something five seconds stale.
        for (int m = 0; m < 2; m++)
            if (m_prev[m].size())
                m_live_err[m] = (m_pod.decode(m_prev[m]) - x).norm() / sigma();
        m_prev[0] = ze;
        m_prev[1] = zl;

        // Catchup: reality has just reached the instant m_catch_i forecast, so
        // the curve gets its next point. the panels hold the last forecast
        // frame throughout, so the plot is the only thing still moving
        if (m_phase == Phase::Catchup) {
            if (m_catch_i < (int)m_pred_esn.size()) {
                const double s = 1.0 / sigma();
                m_curve.push_back(
                    {{s * (m_pod.decode(m_pred_esn[m_catch_i]) - x).norm(),
                      s * (m_pod.decode(m_pred_lstm[m_catch_i]) - x).norm()}});
                resmooth();
            }
            if (++m_catch_i >= (int)m_pred_esn.size()) {
                m_phase = Phase::Live;
                m_next_fc = m_time + DWELL;
            }
            return;
        }

        m_out[0] = m_pod.decode(ze);
        m_out[1] = m_pod.decode(zl);
    }

    // both models have been driven by the live latent all along, so cutting
    // them loose is only a change of what feeds them — no re-washout
    void start_forecast() {
        if (m_phase != Phase::Live)
            return;
        m_pred_esn.clear();
        m_pred_lstm.clear();
        m_curve.clear();
        m_smooth.clear();
        m_catch_i = 0;
        m_fc_accum = 0.0;

        // advance() already RETURNED each model's guess for the next snapshot
        // and staged it as the next input, so the first predict_step() below
        // lands two snapshots out, not one. that staged guess is prediction 0
        // -- without it every point is scored against the truth one snapshot
        // early, which on this shedding period is 13 degrees of phase and puts
        // a flat ~18% under the whole curve
        m_pred_esn.push_back(m_prev[0]);
        m_pred_lstm.push_back(m_prev[1]);
        m_fc_i = 1;
        m_phase = Phase::Forecast;
    }

    // rolls at real time: one model step per SNAP_DT of wall clock, so the
    // prediction evolves at exactly the rate the flow would have
    void forecast_tick(double dt) {
        m_fc_accum += dt;
        while (m_fc_accum >= SNAP_DT && m_fc_i < m_horizon) {
            m_pred_esn.push_back(m_esn->predict_step().col(0));
            m_pred_lstm.push_back(m_lstm->predict_step(m_roll));
            m_fc_accum -= SNAP_DT;
            m_fc_i++;
        }
        m_out[0] = m_pod.decode(m_pred_esn.back());
        m_out[1] = m_pod.decode(m_pred_lstm.back());
        if (m_fc_i >= m_horizon)
            m_phase = Phase::Catchup;
    }

    // ---- drawing ---------------------------------------------------------
    static std::string sfmt(const char *f, ...) {
        char buf[160];
        va_list a;
        va_start(a, f);
        std::vsnprintf(buf, sizeof buf, f, a);
        va_end(a);
        return std::string(buf);
    }

    static Rendering::Color hue(int m) {
        return m == 0 ? Rendering::palette::accent1()
                      : Rendering::palette::accent4();
    }

    static void label(Rendering::Renderer *r, double wx, double wy,
                      const std::string &s, Rendering::Color c, int sz = 16,
                      int line = 0) {
        int sx, sy;
        r->world_to_screen(wx, wy, &sx, &sy);
        r->draw_text(s, sx + 2, sy - 22 - line * 16, sz, c);
    }

    static void init_field(Rendering::FieldView &fv, int nx, int ny) {
        // no colour bar: three panels sharing one ramp would stack three
        // identical bars down the screen edge, and the POD cell above the reel
        // slot already carries one for the same scale
        // no texel fade: window_alpha() tapers in world units instead, which
        // is the only one of the two that lands on the cropped edge
        fv.init(nx, ny,
                {.supersample = SS, .gamma = FIELD_GAMMA, .colorbar = false},
                Rendering::speed_ramp());
        fv.set_scale(0.0, 2.0 * INFLOW, "speed");
    }

    static double freestream_alpha(double u, double v) {
        const double pert = std::hypot(u - INFLOW, v);
        return std::clamp((pert - PERT_MIN) / (PERT_REF - PERT_MIN), 0.0, 1.0);
    }

    // px is where the VISIBLE window starts, so the texture sits offset left by
    // the clipped run. FieldView always blits its full quad, so the cut is made
    // by dropping alpha rather than by resizing anything
    static Vector2d field_origin(double px, double py) {
        return Vector2d(px - CLIP_L * W, py);
    }

    void
    draw_panel(Rendering::Renderer *r, Rendering::FieldView &fv, double px,
               double py, int nx, int ny, double cell,
               const std::function<void(int, int, double &, double &)> &at) {
        const Vector2d fo = field_origin(px, py);
        fv.render(r, fo.x(), fo.y(), cell,
                  Rendering::FieldView::Sample(
                      [&, fo, cell, nx, ny](double wx, double wy, double &val,
                                            double &a) {
                          const double dx = wx - fo.x(), dy = wy - fo.y();
                          const double w = Rendering::window_alpha(
                              dx, dy, CLIP_L * W, 0.0, CLIP_R * W, H, EDGE_PAD);
                          if (w <= 0.0) {
                              a = 0.0;
                              return;
                          }
                          const int i = (int)(dx / cell), j = (int)(dy / cell);
                          if (i < 0 || i >= nx || j < 0 || j >= ny) {
                              a = 0.0;
                              return;
                          }
                          at(i, j, val, a);
                          a *= w;
                      }));
    }

    void obstacle(Rendering::Renderer *r, double px, double py) {
        const Vector2d c = field_origin(px, py) + (m_center - m_fluid.origin());
        r->draw_circle(c.x(), c.y(), RADIUS, Rendering::palette::foreground());
    }

    void draw_model(Rendering::Renderer *r, int m, double px, const char *title,
                    double vmax) {
        if (m_out[m].size() != FSpec::STATE_DIM) {
            label(r, px, PAIR_LABEL_Y, title, hue(m), 16, 1);
            label(r, px, PAIR_LABEL_Y,
                  m_loaded ? "washing in..." : "no checkpoint",
                  Rendering::palette::text_dim(), 13);
            return;
        }

        constexpr int nc = GX * GY;
        const VectorXd &s = m_out[m];
        draw_panel(r, m_panel[m], px, PAIR_Y, GX, GY, GCELL,
                   [&](int i, int j, double &val, double &a) {
                       const int c = i + j * GX;
                       const double u = s[c], v = s[nc + c];
                       val = std::hypot(u, v) / vmax;
                       a = freestream_alpha(u, v);
                   });
        obstacle(r, px, PAIR_Y);

        label(r, px, PAIR_LABEL_Y, title, hue(m), 16, 1);
        switch (m_phase) {
        case Phase::Forecast:
            label(r, px, PAIR_LABEL_Y,
                  sfmt("free-running  +%.1f s", m_fc_i * SNAP_DT), hue(m), 13);
            break;
        case Phase::Catchup:
            label(r, px, PAIR_LABEL_Y,
                  sfmt("held at +%.1f s   err %.0f%%", m_fc_i * SNAP_DT,
                       100.0 * last_err(m)),
                  Rendering::palette::text_dim(), 13);
            break;
        default:
            label(r, px, PAIR_LABEL_Y,
                  sfmt("tracking live   err %.1f%%", 100.0 * m_live_err[m]),
                  Rendering::palette::text_dim(), 13);
            break;
        }
    }

    double last_err(int m) const {
        return m_curve.empty() ? 0.0 : m_curve.back().err[m];
    }

    // frame + panel ground, matching the mode boxes in the POD cell so the two
    // reduced-order frames read as one family. pinned to Layer::Grid, which is
    // what puts it under the content
    void box(Rendering::Renderer *r, double px, double py, double w,
             double h) const {
        Rendering::LayerScope grid(r, Rendering::Layer::Grid);
        const double cx = px + 0.5 * w, cy = py + 0.5 * h;
        const double b = MG_BORDER * h;
        r->draw_rounded_rect(cx, cy, w + 2.0 * b, h + 2.0 * b,
                             Rendering::palette::grid_axis(), 0.0, MG_ROUND);
        r->draw_rounded_rect(cx, cy, w, h, Rendering::palette::background(),
                             0.0, MG_ROUND);
    }

    // error against the horizon it was predicted at. it can only be drawn once
    // the flow has caught back up to that instant, so it fills in left to right
    // during the catch-up while both panels sit frozen above it
    void draw_plot(Rendering::Renderer *r) {
        box(r, PLOT_X0, PLOT_Y, PLOT_W, PLOT_H);

        const double t_max = std::max(1e-6, m_horizon * SNAP_DT);
        const double e_max = plot_ceiling();
        {
            Rendering::LayerScope g(r, Rendering::Layer::Grid);
            for (int k = 1; k < 4; k++) {
                const double y = PLOT_Y + PLOT_H * k / 4.0;
                r->draw_line(PLOT_X0, y, PLOT_X0 + PLOT_W, y, 0.6,
                             Rendering::palette::grid_line());
            }
        }

        label(r, PLOT_X0, PLOT_Y + PLOT_H,
              sfmt("FORECAST ERROR   0 .. %.1f s", t_max),
              Rendering::palette::text_dim(), 13);

        // the axis reads off the two bounding lines, so the scale is stated
        // where it applies rather than in the title
        gutter(r, PLOT_X0 - 0.28, PLOT_Y + PLOT_H,
               sfmt("%.0f%%", 100.0 * e_max), Rendering::palette::text_dim(),
               true);
        gutter(r, PLOT_X0 - 0.28, PLOT_Y, "0%", Rendering::palette::text_dim(),
               true);

        if (m_phase == Phase::Forecast) {
            label(r, PLOT_X0 + 0.2 * PLOT_W, PLOT_Y + 0.15 * PLOT_H,
                  "resolves as the flow catches up",
                  Rendering::palette::text_dim(), 13);
            return;
        }

        if (m_smooth.size() < 2)
            return;
        for (int m = 1; m >= 0; m--)
            trace(r, m, t_max, e_max, 0.2, hue(m));
        draw_final_tags(r, e_max);
    }

    // where each trace ends up, printed off the right end of that trace. two
    // lines a percent apart would otherwise need the reader to match colours,
    // and the tags are nudged apart in SCREEN space -- a world-unit gap that
    // clears at full size collapses into a pile once the cell is scaled into a
    // reel slot
    void draw_final_tags(Rendering::Renderer *r, double e_max) {
        constexpr int FS = 12;
        const int last = (int)m_smooth.size() - 1;

        int sx, sy0, sy1;
        r->world_to_screen(PLOT_X0, 0.0, &sx, &sy0);
        r->world_to_screen(PLOT_X0, 1.0, &sx, &sy1);
        const double px_per_unit = std::abs(sy1 - sy0);
        const double gap =
            px_per_unit > 1e-6 ? (FS + 4) / px_per_unit : 0.4 * PLOT_H;

        struct Tag {
            double y;
            int m;
        };
        Tag tags[2] = {{py_at(last, 0, e_max), 0}, {py_at(last, 1, e_max), 1}};
        if (tags[0].y > tags[1].y)
            std::swap(tags[0], tags[1]);
        tags[1].y = std::max(tags[1].y, tags[0].y + gap);
        const double over = tags[1].y - (PLOT_Y + PLOT_H);
        if (over > 0.0)
            for (Tag &t : tags)
                t.y -= over;

        for (const Tag &t : tags)
            gutter(r, PLOT_X0 + PLOT_W + 0.28, t.y,
                   sfmt("%.1f%%", 100.0 * m_smooth[last].err[t.m]), hue(t.m),
                   false);
    }

    // text vertically centred on a world y, hung off one side of the plot
    static void gutter(Rendering::Renderer *r, double wx, double wy,
                       const std::string &s, Rendering::Color c,
                       bool right_align, int fs = 12) {
        Rendering::LayerScope layer(r, Rendering::Layer::Text);
        int sx, sy;
        r->world_to_screen(wx, wy, &sx, &sy);
        r->draw_text(s, right_align ? sx - r->measure_text(s, fs) : sx,
                     sy - fs / 2, fs, c);
    }

    static Vector2d catmull(const Vector2d &p0, const Vector2d &p1,
                            const Vector2d &p2, const Vector2d &p3, double u) {
        const double u2 = u * u, u3 = u2 * u;
        return 0.5 * (2.0 * p1 + (p2 - p0) * u +
                      (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2 +
                      (3.0 * p1 - p0 - 3.0 * p2 + p3) * u3);
    }

    // Catmull-Rom through the samples, emitted as short segments. Drawing the
    // samples as one raw polyline leaves a visible cap wherever two segments
    // meet at a sharp angle. SUB is kept low on purpose: every segment is its
    // own shader draw, and the finished curve is redrawn each frame for the
    // whole live dwell, which at SUB 6 across three traces was ~900 draws a
    // frame and visibly dragged the demo down until the next rollout cleared it
    void trace(Rendering::Renderer *r, int m, double t_max, double e_max,
               double th, Rendering::Color c) {
        constexpr int SUB = 3;
        const int n = (int)m_smooth.size();
        auto P = [&](int i) {
            const int k = std::clamp(i, 0, n - 1);
            return Vector2d(px_at(k, t_max), py_at(k, m, e_max));
        };

        Vector2d prev = P(0);
        for (int i = 0; i < n - 1; i++)
            for (int s = 1; s <= SUB; s++) {
                const Vector2d q = catmull(P(i - 1), P(i), P(i + 1), P(i + 2),
                                           (double)s / SUB);
                r->draw_smooth_line(prev.x(), prev.y(), q.x(), q.y(), th, c);
                prev = q;
            }
    }

    // the wake slides about a cell and a half per snapshot, so how well a fixed
    // basis lands on it beats at a period of a few samples. that ripple belongs
    // to the grid rather than to either forecaster, so it is rolled out before
    // the traces are drawn
    void resmooth() {
        constexpr int R = 2;
        const int n = (int)m_curve.size();
        m_smooth.assign(n, Point{});
        for (int i = 0; i < n; i++) {
            const int lo = std::max(0, i - R), hi = std::min(n - 1, i + R);
            for (int m = 0; m < 2; m++) {
                double s = 0.0;
                for (int k = lo; k <= hi; k++)
                    s += m_curve[k].err[m];
                m_smooth[i].err[m] = s / (hi - lo + 1);
            }
        }
    }

    // a fixed ceiling either flattens a good rollout against the floor or clips
    // a bad one against the lid, and which of the two a cycle turns out to be
    // is not knowable until reality has caught up with it
    double plot_ceiling() const {
        double m = ERR_FLOOR;
        for (const Point &p : m_smooth)
            m = std::max({m, p.err[0], p.err[1]});
        return std::min(1.05 * m, 2.0);
    }

    static double px_at(int i, double t_max) {
        return PLOT_X0 + PLOT_W * (i + 1) * SNAP_DT / t_max;
    }
    double py_at(int i, int m, double e_max) const {
        return PLOT_Y +
               PLOT_H * std::clamp(m_smooth[i].err[m] / e_max, 0.0, 1.0);
    }

    void draw_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("KARMAN ESN vs LSTM", Rendering::palette::accent2());
        if (!m_loaded) {
            hud.small_text("no checkpoint found at",
                           Rendering::palette::accent1());
            hud.small_text(m_path.c_str(), Rendering::palette::text_dim());
            hud.small_text("run tools/train_forecast_demo",
                           Rendering::palette::text_dim());
            return;
        }
        const char *ph = m_phase == Phase::Forecast  ? "FREE-RUNNING"
                         : m_phase == Phase::Catchup ? "CATCH-UP"
                                                     : "LIVE";
        hud.line(Rendering::palette::text(), "t %.1f s   [%s]", m_time, ph);
        hud.line(Rendering::palette::text_dim(), "coefficients driven %d",
                 m_driven);
        hud.line(Rendering::palette::text(), "horizon %d  (%.1f s)", m_horizon,
                 m_horizon * SNAP_DT);
        hud.line(hue(0), "ESN   live %.1f%%  fc %.0f%%", 100.0 * m_live_err[0],
                 100.0 * last_err(0));
        hud.line(hue(1), "LSTM  live %.1f%%  fc %.0f%%", 100.0 * m_live_err[1],
                 100.0 * last_err(1));
        hud.separator();
        hud.small_text("[Space] forecast now  [Up/Down] horizon  [R] reset",
                       Rendering::palette::text_dim());
    }

    // ---- state -----------------------------------------------------------
    Fluid::StableFluidSolver m_fluid{(unsigned)ROWS,
                                     (unsigned)COLS,
                                     CELL,
                                     FSpec::VISC,
                                     0.0,
                                     Vector2d(FSpec::OX, FSpec::OY)};
    Vector2d m_center = Vector2d::Zero();
    AI::SnapshotRecorder m_recorder;
    Rendering::FieldView m_field, m_panel[2];

    AI::POD m_pod;
    std::optional<AI::ESN> m_esn;
    std::optional<AI::LSTM> m_lstm;
    AI::LSTM::Rollout m_roll;
    std::string m_path;
    bool m_loaded = false;

    VectorXd m_out[2]; // decoded fields on display, [esn, lstm]
    std::vector<VectorXd> m_pred_esn, m_pred_lstm; // the frozen rollout
    std::vector<Point> m_curve, m_smooth;
    VectorXd m_prev[2]; // last open-loop guess, scored when its instant lands
    double m_live_err[2] = {0.0, 0.0};

    Phase m_phase = Phase::Live;
    double m_time = 0.0, m_next_fc = FIRST_FC, m_fc_accum = 0.0;
    double m_sig_sum = 0.0; // running |x - mean|^2, the error reference
    int m_sig_n = 0;
    double m_snap_accum = 0.0; // sim time owed to the next snapshot
    int m_driven = 0;
    int m_horizon = HORIZON, m_fc_i = 0, m_catch_i = 0;
};

} // namespace manifold::Demo
