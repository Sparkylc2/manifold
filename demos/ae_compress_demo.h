#pragma once

#include "forecast_demo_spec.h"
#include "manifold/renderer/body_visuals.h"
#include "manifold/renderer/theme.h"

#include <manifold/ai/autoencoder.h>
#include <manifold/ai/snapshot_recorder.h>
#include <manifold/fluid/stable_fluid_solver.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <string>

namespace manifold::Demo {

// repeated identically in forecast_karman_demo.h, which is legal and is what
// lets the story include both cells in one translation unit
namespace FSpec = ForecastSpec;

using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;

// The compression on its own, with nothing forecasting anything: the live field
// goes into a pretrained dense autoencoder, out the other side, and the only
// thing that crossed between them is the row of five numbers drawn in the
// middle. The funnel is the whole argument -- 40 000 numbers in, five across
// the waist, 40 000 back out -- so it is drawn literally rather than described.
class AECompressDemo : public DemoBase {
  public:
    static constexpr int COLS = FSpec::COLS, ROWS = FSpec::ROWS;
    static constexpr double CELL = FSpec::CELL;
    static constexpr double W = FSpec::W, H = FSpec::H;
    static constexpr double INFLOW = FSpec::INFLOW, RADIUS = FSpec::RADIUS;
    static constexpr double SNAP_DT = FSpec::SNAP_DT;
    static constexpr int CODE = FSpec::AE_CODE;
    static constexpr int STATE_DIM = FSpec::STATE_DIM;

    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int SS = 2;
    // world-space taper on the visible window. this cell crops on BOTH axes, so
    // without it all four sides of each panel end on a hard line
    static constexpr double EDGE_PAD = 2.0;
    static constexpr double FIELD_GAMMA = 0.29;

    // ---- layout, world units ----
    // cropped on both axes here, not just across: stacking two panels makes the
    // cell tall and narrow, and the freestream margins top and bottom are the
    // cheapest height to give back
    static constexpr double CLIP_L = 0.15, CLIP_R = 1.00;
    static constexpr double CLIP_B = 0.15, CLIP_T = 0.85;
    static constexpr double PW = W * (CLIP_R - CLIP_L);
    static constexpr double PH = H * (CLIP_T - CLIP_B);
    // Captions anchor this far inside their band rather than on its edge. The
    // crop already trims the uniform freestream, and the edge taper fades what
    // is left, so a caption hung on the band edge floats clear of the picture.
    static constexpr double LABEL_INSET = 0.9;
    // World slack the bounds need either end. Captions are drawn at a fixed
    // PIXEL offset from their anchor, so they escape the bounds by an amount
    // that grows as the cell is drawn smaller -- and bounds are what the reel
    // slot fits, so that overflow lands in the neighbouring cell's band. This
    // covers the tallest caption (title + subtitle, 38 px) at reel scale.
    static constexpr double LABEL_PAD = 1.8;

    // The extra height the cell was given goes entirely into the funnel: the
    // two panels keep PH, and only the span from FUN_TOP to FUN_BOT stretches,
    // which is also what gives the encoder and decoder halves room to be named.
    // FUN_BOT is the one number that sets the funnel's height -- everything
    // below derives from it, so shrinking it really does shrink the funnel and
    // pulls the reconstruction panel and the cell bounds up with it. WAIST_H is
    // the waist HALF-HEIGHT, so lowering that lengthens the legs rather than
    // shortening the funnel.
    static constexpr double FUN_TOP = -0.4; // encoder mouth
    static constexpr double FUN_BOT = -5.6; // decoder mouth
    static constexpr double DOT_Y = 0.5 * (FUN_TOP + FUN_BOT); // the waist
    static constexpr double WAIST_H = 1.1; // waist half-height
    // Each half named between its own pair of legs, where the middle is empty.
    // Biased toward its own mouth rather than sat at the midpoint, which put
    // both names right up against the caption and the dot row.
    static constexpr double NAME_T = 0.30; // 0 = at the mouth, 0.5 = midpoint
    static constexpr double ENC_Y =
        FUN_TOP + NAME_T * ((DOT_Y + WAIST_H) - FUN_TOP);
    static constexpr double DEC_Y =
        FUN_BOT + NAME_T * ((DOT_Y - WAIST_H) - FUN_BOT);
    // the reconstruction's caption hangs BELOW its panel rather than above it:
    // above is where the decoder mouth is, and at reel scale a label is a fixed
    // pixel height while the funnel shrinks with the cell, so no world-space
    // gap clears it at every size
    static constexpr double RECON_GAP = 0.6; // decoder mouth -> lower panel
    static constexpr double RECON_TOP = FUN_BOT - RECON_GAP;
    static constexpr double RECON_Y = RECON_TOP - PH;

    static constexpr double DOT_SP = 0.85;
    static constexpr double DOT_R = 0.30;
    static constexpr double WAIST =
        (CODE - 1) * DOT_SP; // dot row, centre to centre

    // the dot row and its caption move as one block, both centred on the panel:
    // CODE_Y is the row, CODE_LABEL_DY the caption's fixed offset above it
    static constexpr double CODE_LABEL_DY = 1.5;
    // the pair straddles the waist, so the block stays centred whatever the
    // caption offset is
    static constexpr double CODE_Y = DOT_Y - 0.5 * CODE_LABEL_DY;

    // Funnel in x, as half-widths about the panel centre. The mouths start well
    // inside the panel edges and the inner ends stop well clear of the caption,
    // so the taper reads as a funnel rather than as two full-width diagonals.
    static constexpr double FUN_MOUTH_HW = 5.8;
    static constexpr double FUN_WAIST_HW = 3.3;
    static constexpr double FUN_TH = 0.30; // bar thickness, world units

    // the checkpoint was fitted to a street recorded from t = 20 s on, so the
    // cell has to be at least that developed before its error means anything --
    // at t = 12 the same weights read 33% instead of 8%
    static constexpr double WARM = 20.0;

    // the cell publishes its own extent so the reel slot cannot drift out of
    // step with the geometry above it
    static constexpr double BOUND_X0 = 0.0, BOUND_X1 = PW;
    static constexpr double BOUND_Y1 = PH - LABEL_INSET + LABEL_PAD;
    static constexpr double BOUND_Y0 = RECON_Y + LABEL_INSET - LABEL_PAD;

    const char *name() const override { return "Karman Autoencoder"; }
    double default_cam_x() const override { return 0.5 * PW; }
    double default_cam_y() const override { return -2.8; }
    double default_cam_zoom() const override { return 30.0; }

    void initialize() override {
        m_fluid.clear();
        m_fluid.set_channel(INFLOW);
        m_center = FSpec::center();
        m_fluid.set_circle_obstacle(m_center, RADIUS);
        m_recorder = AI::SnapshotRecorder(COLS, ROWS, CELL, m_fluid.origin(), 1,
                                          1, 0.0, 2);

        m_time = 0.0;
        m_accum = 0.0;
        m_recon.resize(0);
        m_code.resize(0);
        m_code_scale.resize(0);
        m_err = 0.0;

        m_ae.build(STATE_DIM, FSpec::ae_layers(), CODE, 1);
        m_path = FSpec::ae_weights_path();
        m_loaded = FSpec::load_ae(m_path, m_ae, m_mean, m_sigma);
        if (!m_loaded)
            std::printf("[ae] no checkpoint at %s -- run train_forecast_demo\n",
                        m_path.c_str());

        init_field(m_field);
        init_field(m_recon_field);
    }

    void process(double dt) override {
        m_fluid.advance(dt);
        m_time += dt;

        // the encode/decode pair is two dense passes over 40k inputs, which is
        // not worth doing every frame for a picture that changes this slowly
        m_accum += dt;
        if (m_accum < SNAP_DT - 1e-9)
            return;
        m_accum = 0.0;
        if (!m_loaded)
            return;

        const VectorXd x = m_recorder.sample_state(m_fluid);
        m_code = m_ae.encode(x);
        // each latent lives on its own scale, and normalising the row against
        // whichever is largest leaves the rest as invisible specks. a running
        // per-dimension max says "how far is this one from its own extreme",
        // which is the thing the row is meant to show
        if (m_code_scale.size() != CODE)
            m_code_scale = VectorXd::Constant(CODE, 1e-6);
        m_code_scale = m_code_scale.cwiseMax(m_code.cwiseAbs());
        m_recon = m_ae.decode(m_code);
        m_err = m_sigma > 1e-9 ? (m_recon - x).norm() / m_sigma : 0.0;
    }

    void render(Rendering::Renderer *r) override {
        render_cell(r);
        draw_hud(r);
    }

    void render_cell(Rendering::Renderer *r) override {
        const double vmax = 2.0 * INFLOW;
        const Vector2d o = m_fluid.origin();

        draw_panel(r, m_field, 0.0, [&](int i, int j, double &val, double &a) {
            Vector2d v;
            m_fluid.velocity_at(
                Vector2d(o.x() + (i + 0.5) * CELL, o.y() + (j + 0.5) * CELL),
                &v, Fluid::Interp::Cubic);
            val = v.norm() / vmax;
            a = freestream_alpha(v.x(), v.y());
        });
        obstacle(r, 0.0);
        label(r, 0.0, PH - LABEL_INSET, "LIVE FLOW",
              Rendering::palette::accent2(), 16, 1);
        label(r, 0.0, PH - LABEL_INSET,
              sfmt("%s numbers in", grouped(STATE_DIM).c_str()),
              Rendering::palette::text_dim(), 13);

        draw_funnel(r);
        draw_code(r);

        if (m_recon.size() == STATE_DIM) {
            constexpr int nc = COLS * ROWS;
            draw_panel(r, m_recon_field, RECON_Y,
                       [&](int i, int j, double &val, double &a) {
                           const int c = i + j * COLS;
                           const double u = m_recon[c], v = m_recon[nc + c];
                           val = std::hypot(u, v) / vmax;
                           a = freestream_alpha(u, v);
                       });
            obstacle(r, RECON_Y);
        }
        below(r, 0.0, RECON_Y + LABEL_INSET, "RECONSTRUCTION",
              Rendering::palette::accent1(), 16, 0);
        below(r, 0.0, RECON_Y + LABEL_INSET,
              m_loaded ? sfmt("%s numbers back out   err %.0f%%",
                              grouped(STATE_DIM).c_str(), 100.0 * m_err)
                       : std::string("no checkpoint"),
              Rendering::palette::text_dim(), 13, 1);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
    }

  private:
    // ---- drawing ---------------------------------------------------------
    static std::string sfmt(const char *f, ...) {
        char buf[160];
        va_list a;
        va_start(a, f);
        std::vsnprintf(buf, sizeof buf, f, a);
        va_end(a);
        return std::string(buf);
    }

    // "40 000" rather than "40000" -- the number is the point of the frame and
    // it has to be read at a glance, not counted
    static std::string grouped(long v) {
        std::string d = std::to_string(v), out;
        for (int i = 0; i < (int)d.size(); i++) {
            if (i && (d.size() - i) % 3 == 0)
                out += ' ';
            out += d[i];
        }
        return out;
    }

    static void label(Rendering::Renderer *r, double wx, double wy,
                      const std::string &s, Rendering::Color c, int sz = 16,
                      int line = 0) {
        int sx, sy;
        r->world_to_screen(wx, wy, &sx, &sy);
        r->draw_text(s, sx + 2, sy - 22 - line * 16, sz, c);
    }

    // same as label(), hung under the anchor instead of over it
    static void below(Rendering::Renderer *r, double wx, double wy,
                      const std::string &s, Rendering::Color c, int sz,
                      int line) {
        int sx, sy;
        r->world_to_screen(wx, wy, &sx, &sy);
        r->draw_text(s, sx + 2, sy + 8 + line * 18, sz, c);
    }

    static void centred(Rendering::Renderer *r, double wx, double wy,
                        const std::string &s, Rendering::Color c, int sz) {
        Rendering::LayerScope t(r, Rendering::Layer::Text);
        int sx, sy;
        r->world_to_screen(wx, wy, &sx, &sy);
        r->draw_text(s, sx - r->measure_text(s, sz) / 2, sy - sz / 2, sz, c);
    }

    static void init_field(Rendering::FieldView &fv) {
        // no texel fade: window_alpha() tapers in world units instead, which is
        // the only one of the two that lands on the cropped edges
        fv.init(COLS, ROWS,
                {.supersample = SS, .gamma = FIELD_GAMMA, .colorbar = false},
                Rendering::speed_ramp());
        fv.set_scale(0.0, 2.0 * INFLOW, "speed");
    }

    static double freestream_alpha(double u, double v) {
        const double pert = std::hypot(u - INFLOW, v);
        return std::clamp((pert - PERT_MIN) / (PERT_REF - PERT_MIN), 0.0, 1.0);
    }

    // py is where the VISIBLE band starts; FieldView always blits its whole
    // quad, so both cuts are made by dropping alpha rather than by resizing
    static Vector2d field_origin(double py) {
        return Vector2d(-CLIP_L * W, py - CLIP_B * H);
    }

    void
    draw_panel(Rendering::Renderer *r, Rendering::FieldView &fv, double py,
               const std::function<void(int, int, double &, double &)> &at) {
        const Vector2d fo = field_origin(py);
        fv.render(r, fo.x(), fo.y(), CELL,
                  Rendering::FieldView::Sample(
                      [&, fo](double wx, double wy, double &val, double &a) {
                          const double dx = wx - fo.x(), dy = wy - fo.y();
                          const double win = Rendering::window_alpha(
                              dx, dy, CLIP_L * W, CLIP_B * H, CLIP_R * W,
                              CLIP_T * H, EDGE_PAD);
                          if (win <= 0.0) {
                              a = 0.0;
                              return;
                          }
                          const int i = (int)(dx / CELL), j = (int)(dy / CELL);
                          if (i < 0 || i >= COLS || j < 0 || j >= ROWS) {
                              a = 0.0;
                              return;
                          }
                          at(i, j, val, a);
                          a *= win;
                      }));
    }

    void obstacle(Rendering::Renderer *r, double py) {
        const Vector2d c = field_origin(py) + (m_center - m_fluid.origin());
        r->draw_circle(c.x(), c.y(), RADIUS, Rendering::palette::foreground());
    }

    // the encoder and decoder as one hourglass: everything the panel above
    // holds has to pass through the waist to reach the panel below
    // draw_body_bar centres the bar on the position it is given, so a leg is
    // placed at the MIDPOINT of the span it covers -- handing it an endpoint
    // puts half the bar out beyond that end
    void leg(Rendering::Renderer *r, double xa, double ya, double xb,
             double yb) {
        const Vector2d d(xb - xa, yb - ya);
        Rendering::draw_body_bar(r, 0.5 * (xa + xb), 0.5 * (ya + yb), d.norm(),
                                 FUN_TH, std::atan2(d.y(), d.x()),
                                 {.border = Rendering::palette::foreground(),
                                  .border_width = 0.0,
                                  .fill = Rendering::palette::text_dim(),
                                  .show_center = false,
                                  .show_shadow = true});
    }

    // the encoder and decoder as one hourglass, symmetric about the same centre
    // the dots and caption use: everything the panel above holds has to pass
    // through the waist to reach the panel below
    void draw_funnel(Rendering::Renderer *r) {
        Rendering::LayerScope g(r, Rendering::Layer::Grid);
        const double cx = code_cx();
        const double wy0 = DOT_Y + WAIST_H, wy1 = DOT_Y - WAIST_H;

        for (const double s : {-1.0, 1.0}) {
            leg(r, cx + s * FUN_MOUTH_HW, FUN_TOP, cx + s * FUN_WAIST_HW, wy0);
            leg(r, cx + s * FUN_WAIST_HW, wy1, cx + s * FUN_MOUTH_HW, FUN_BOT);
        }

        centred(r, cx, ENC_Y, "ENCODER", Rendering::palette::text_dim(), 13);
        centred(r, cx, DEC_Y, "DECODER", Rendering::palette::text_dim(), 13);
    }

    // one centre for the funnel, the dots and the caption, so they cannot drift
    // apart: the middle of the drawn panel
    static constexpr double code_cx() { return 0.5 * PW; }

    static Rendering::Color node_color(double t) {
        t = std::clamp(t, -1.0, 1.0);
        const Rendering::Color mid = Rendering::palette::background();
        return t >= 0 ? Rendering::color_lerp(mid,
                                              Rendering::palette::accent1(), t)
                      : Rendering::color_lerp(
                            mid, Rendering::palette::accent4(), -t);
    }

    void draw_code(Rendering::Renderer *r) {
        const double cx = code_cx();
        const double x0 = cx - 0.5 * WAIST;
        const bool have = m_code.size() == CODE && m_code_scale.size() == CODE;

        for (int i = 0; i < CODE; i++) {
            double t = have ? m_code[i] / std::max(m_code_scale[i], 1e-9) : 0.0;
            // compressive: a latent sitting at a third of its range should read
            // as clearly present, not as a faint smudge
            t = t >= 0.0 ? std::sqrt(t) : -std::sqrt(-t);

            Rendering::draw_body_disk(
                r, x0 + i * DOT_SP, CODE_Y, DOT_R, 0.0,
                {.show_center = false,
                 .show_shadow = false,
                 .fill = node_color(t),
                 .border_width = 0.05,
                 .border = Rendering::palette::foreground()});
        }

        centred(r, cx, CODE_Y + CODE_LABEL_DY, sfmt("%d numbers", CODE),
                Rendering::palette::text_dim(), 13);
    }

    void draw_hud(Rendering::Renderer *r) {
        Rendering::HUDPanel hud(r, hud_x(r), 12);
        hud.title("KARMAN AUTOENCODER", Rendering::palette::accent2());
        if (!m_loaded) {
            hud.small_text("no checkpoint found at",
                           Rendering::palette::accent1());
            hud.small_text(m_path.c_str(), Rendering::palette::text_dim());
            hud.small_text("run tools/train_forecast_demo",
                           Rendering::palette::text_dim());
            return;
        }
        hud.line(Rendering::palette::text(), "t %.1f s", m_time);
        hud.line(Rendering::palette::text(), "%s -> %d -> %s",
                 grouped(STATE_DIM).c_str(), CODE, grouped(STATE_DIM).c_str());
        hud.line(Rendering::palette::accent3(), "reconstruction err %.1f%%",
                 100.0 * m_err);
        hud.separator();
        hud.small_text("[R] reset", Rendering::palette::text_dim());
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
    Rendering::FieldView m_field, m_recon_field;

    AI::Autoencoder m_ae;
    VectorXd m_mean, m_recon, m_code;
    VectorXd m_code_scale;
    double m_sigma = 1.0, m_err = 0.0;
    std::string m_path;
    bool m_loaded = false;

    double m_time = 0.0, m_accum = 0.0;
};

} // namespace manifold::Demo
