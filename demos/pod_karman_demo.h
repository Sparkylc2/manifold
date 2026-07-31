#pragma once

#include "manifold/ai/pod.h"
#include "manifold/ai/snapshot_recorder.h"

#include "manifold/renderer/body_visuals.h"
#include "manifold/renderer/interpolation.h"
#include "manifold/renderer/theme.h"
#include <Eigen/Dense>
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
#include <functional>
#include <future>
#include <string>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;
using ArrayXd = Eigen::ArrayXd;
using MatrixXd = Eigen::MatrixXd;

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

    static constexpr double PERT_MIN = 0.05 * INFLOW;
    static constexpr double PERT_REF = 0.60 * INFLOW;
    static constexpr int FADE_PX = 18;

    static constexpr int N_MODES = 8;
    // shedding POD modes come out in near-degenerate quadrature pairs: sigma_2k
    // ~= sigma_2k+1 and the two are one structure a quarter wavelength apart.
    // they share a hue, and the mix below weights them as one family
    static constexpr int PAIR = 2;
    static constexpr int N_FAM = N_MODES / PAIR;
    static constexpr int MODE_COLS = 2;
    static constexpr int N_CAP = 200;
    static constexpr uint STRIDE = 3;
    static constexpr double TRANSIENT = 2.0;
    static constexpr int MIN_SNAPS = 12;
    static constexpr int RECOMPUTE_EVERY = 60;

    static constexpr double GAP = 1.4;
    static constexpr double MCELL = CELL * 0.48;
    static constexpr double MW = COLS * MCELL;
    static constexpr double MH = ROWS * MCELL;
    static constexpr double MGAP = 0.72;
    static constexpr double VGAP = 2.4;

    // compact layout, used only by render_cell()
    static constexpr int CELL_MODES = 4;
    static constexpr double CLIP_L = 0.15, CLIP_R = 1.00;
    static constexpr double CVIS = CLIP_R - CLIP_L;
    static constexpr double CW = W * CVIS;
    static constexpr double CGAP = 0.9;  // live | reconstruction
    static constexpr double CVGAP = 1.9; // pair -> modes
    static constexpr double CMGAP = 1.2; // between mode slots (holds the +)
    static constexpr double CBAR_W = 0.38, CBAR_GAP = 0.30;
    // a mode slot is panel + gap + bar; four of them span the pair above
    static constexpr double CMW =
        (2.0 * CW + CGAP - 3.0 * CMGAP - CELL_MODES * (CBAR_GAP + CBAR_W)) /
        CELL_MODES;
    static constexpr double CMCELL = CMW / (CVIS * COLS);
    static constexpr double CMH = ROWS * CMCELL;
    static constexpr double CMSLOT = CMW + CBAR_GAP + CBAR_W;

    static constexpr double REVEAL_STEP = 2.4;
    static constexpr double REVEAL_HOLD = 3.6;
    static constexpr double HILITE_DUR = 0.45; // border fade-IN, then it holds
    static constexpr double BAR_IN = 0.25;
    static constexpr double HILITE_M = 0.10; // hilite inset, also the bar's
    static constexpr double MODE_GAMMA = 0.5;
    static constexpr double BAR_ROUND = 0.45; // 1.0 would be a full capsule
    // the harmonic pair holds ~3% of the leader's energy, which is a bar two
    // pixels tall on a phone. compressed and floored so it reads, at the cost
    // of the bars no longer being linear in KE -- the "%.1f%% KE" label beside
    // each one is what carries the actual number
    static constexpr double BAR_GAMMA = 0.95;
    static constexpr double BAR_MIN = 0.04;
    static constexpr double BAR_TAU = 0.30; // ease onto a recomputed split
    static constexpr double FIELD_GAMMA = 0.29;
    // thicknesses track RaylibRenderer::draw_grid (1.0 line / 2.0 axis), with
    // the lines pulled under it and a finer subdivision added below those
    static constexpr double MG_PITCH = 0.26; // major pitch, fraction of height
    static constexpr int MG_SUB = 3;         // minor cells per major cell
    static constexpr double MG_TH_MAJ = 0.8;
    static constexpr double MG_TH_MIN = 0.45;
    static constexpr double MG_TH_BOX = 2.0;
    static constexpr double MG_MIN_A = 0.45; // minor tint, bg -> grid_line
    static constexpr double MG_ROUND = 0.10;
    static constexpr double MG_BORDER = 0.012; // frame width, fraction of h

    static constexpr double AMP_TAU = 0.08;    // rides out recompute jumps
    static constexpr double MODE_FLOOR = 0.22; // opacity of the far-sign lobes
    static constexpr double SIGN_GAIN = 2.2;   // contrast between the signs

    // modal-mix recolouring of the reconstruction. OFF by default and not used
    // by the reel: on this flow the family envelopes overlap at r = 0.95 with
    // centroids ~1 cell apart, so there is no spatial territory to colour and
    // the result is a flatter monochrome version of the speed ramp. kept behind
    // [M] because the machinery is the interesting part, not the picture
    static constexpr double MIX_SHARP = 1.8;  // >1 pulls toward the leader
    static constexpr double MIX_WHITE = 0.42; // top-end wash toward white
    static constexpr double MIX_KNEE = 0.62;  // where hue -> washed hue

    void set_auto_reveal(bool on) { m_auto_reveal = on; }
    void set_modal_mix(bool on) { m_modal_mix = on; }

    const char *name() const override { return "Karman POD"; }

    double default_cam_y() const override { return 5.0; }
    double default_cam_zoom() const override { return 16.0; }

    void initialize() override {
        // drain any in-flight solve before tearing state down
        if (m_pod_job.valid())
            m_pod_job.wait();
        m_job_running = false;
        m_have_pending = false;
        m_reveal_step = -1;
        m_reveal_edge = false;

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

        init_field(m_field, false);
        init_field(m_recon_field, false);
        for (auto &mv : m_mode_views) {
            mv.init(COLS, ROWS,
                    {.supersample = 1,
                     .edge_fade_px = 6,
                     .gamma = MODE_GAMMA,
                     .colorbar = false},
                    Rendering::speed_ramp());
            mv.set_scale(0.0, 1.0, "mode");
        }
        m_coef.fill(0.0);
        m_coef_rms.fill(1.0);
        m_amp.fill(0.0);
        m_bar.fill(0.0);
        m_fam_amp.fill(0.0);
        m_num_fam = 0;
        m_mix_n = 0;
        m_amp_primed = false;
        m_bar_primed = false;

        m_reveal_t = 0.0;
        m_ke_plot.configure("KE err %", Rendering::palette::accent3(), 600);
        m_ke_plot.clear();
    }

    void process(double dt) override {
        m_fluid.advance(dt); // sim never blocks on the SVD
        m_time += dt;
        if (m_auto_reveal) {
            m_reveal_t += dt;
            m_rank = revealed();
        }
        m_recorder.maybe_capture(m_fluid, m_time);
        m_frame++;

        // tracked every frame, not only when a solve is waiting, so the edge is
        // the real reveal boundary rather than whenever we happened to look
        const int step = m_auto_reveal ? (int)(cycle_t() / REVEAL_STEP) : 0;
        m_reveal_edge = step != m_reveal_step;
        m_reveal_step = step;

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
            m_pending = m_pod_job.get();
            m_have_pending = true;
            m_job_running = false;
        }

        // a solve lands whenever it finishes, which is mid-hold more often than
        // not. alignment takes most of the jump out, but the mean flow and the
        // singular values still shift a little, so the swap is parked until the
        // reveal brings in the next mode and hides whatever is left of it
        if (m_have_pending &&
            (!m_have_pod || !m_auto_reveal || m_reveal_edge)) {
            m_pod = std::move(m_pending);
            cache_modes();
            m_have_pod = true;
            m_have_pending = false;
        }

        if (m_have_pod)
            update_reconstruction(dt);
    }

    void render(Rendering::Renderer *r) override {
        draw_grid(r);
        draw_wide(r);

        draw_hud(r);

        // KE trace, bottom-left (clear of the top-right colourbar)
        m_ke_plot.render(r, hud_x(r), r->screen_height() - 92, 280, 80);
    }

    void draw_wide(Rendering::Renderer *r) {
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
    }

    void render_cell(Rendering::Renderer *r) override {
        const Vector2d o = m_fluid.origin();
        const double vmax = 2.0 * INFLOW;
        constexpr int nc = COLS * ROWS;

        const double lx = o.x(), ly = o.y();
        const double rx = lx + CW + CGAP;

        // live, left
        draw_panel(r, m_field, lx, ly, CELL,
                   [&](int i, int j, double &val, double &a) {
                       Vector2d v;
                       m_fluid.velocity_at(Vector2d(o.x() + (i + 0.5) * CELL,
                                                    o.y() + (j + 0.5) * CELL),
                                           &v, Fluid::Interp::Cubic);
                       val = v.norm() / vmax;
                       a = freestream_alpha(v.x(), v.y());
                   });
        const Vector2d lo = panel_origin(lx, ly, CELL);
        r->draw_circle((lo + (m_center - o)).x(), (lo + (m_center - o)).y(),
                       RADIUS, Rendering::palette::foreground());
        label(r, lx, ly + H, "LIVE", Rendering::palette::accent2());

        // reconstruction, right
        if (m_have_pod && m_recon.size() != 0) {
            if (m_modal_mix) {
                prime_mix();
                draw_panel_mix(
                    r, m_recon_field, rx, ly, CELL,
                    [&](int i, int j, Rendering::Color &col, double &a) {
                        const int c = i + j * COLS;
                        const double u = m_recon[c], v = m_recon[nc + c];
                        col = mix_color(c, std::hypot(u, v) / vmax);
                        a = freestream_alpha(u, v);
                    });
            } else {
                draw_panel(r, m_recon_field, rx, ly, CELL,
                           [&](int i, int j, double &val, double &a) {
                               const int c = i + j * COLS;
                               const double u = m_recon[c], v = m_recon[nc + c];
                               val = std::hypot(u, v) / vmax;
                               a = freestream_alpha(u, v);
                           });
            }
            const Vector2d ro = panel_origin(rx, ly, CELL) + (m_center - o);
            r->draw_circle(ro.x(), ro.y(), RADIUS,
                           Rendering::palette::foreground());
            label(r, rx, ly + H, sfmt("RECONSTRUCTION  rank %d", rank()),
                  Rendering::palette::accent2());
        }

        if (!m_have_pod)
            return;

        // modes in one row underneath, each with a vertical KE bar beside it
        const double my = ly - CVGAP - CMH;
        const double sc = CMCELL / CELL;
        const int n = std::min(CELL_MODES, m_num_shown);

        // normalised against the eased energies too, otherwise the reference
        // jumps on the swap and every bar twitches to compensate
        double e_ref = 1e-9;
        for (int k = 0; k < n; k++)
            e_ref = std::max(e_ref, m_bar[k]);

        const int shown = m_auto_reveal ? std::min(n, revealed()) : n;
        const double age = newest_age();

        for (int k = 0; k < shown; k++) {
            const double px = lx + k * (CMSLOT + CMGAP);
            const double scale = m_mode_scale[k];
            const bool newest = m_auto_reveal && k == shown - 1;

            // colour is the frozen mode shape through the shared speed ramp, so
            // a mode panel is the same kind of picture as the fields above it.
            // a_k(t) rides on the opacity, per texel, so alternate lobes fade
            // against each other and the pattern reads as travelling
            const double amp = m_amp[k];
            const double us = m_sign_scale[k];
            constexpr int nc = COLS * ROWS;

            draw_mode_box(r, px, my, CMW, CMH);

            draw_panel(
                r, m_mode_views[k], px, my, CMCELL,
                [&, k, scale, amp, us](int i, int j, double &val, double &a) {
                    const int c = i + j * COLS;
                    const VectorXd &m = m_mode_data[k];
                    val = std::hypot(m[c], m[nc + c]) / scale;
                    a = std::clamp(val * sign_alpha(amp, m[c] / us), 0.0, 1.0);
                });

            const Vector2d mo =
                panel_origin(px, my, CMCELL) + (m_center - o) * sc;
            r->draw_circle(mo.x(), mo.y(), RADIUS * sc,
                           Rendering::palette::foreground());

            const double e = m_bar[k];
            label(r, px, my + CMH, sfmt("MODE %d", k + 1), mode_hue(k), 14);
            label(r, px, my + CMH, sfmt("%.1f%% KE", 100.0 * e),
                  Rendering::palette::text_dim(), 12, 1);

            const double grow = newest ? ease_out(age / BAR_IN) : 1.0;
            draw_ke_bar(r, px + CMW + CBAR_GAP, my, grow * bar_frac(e / e_ref),
                        k);

            // the per-mode hilite box is superseded by draw_mode_box, which
            // frames every panel rather than only the revealed ones:
            //   const double kage =
            //       m_auto_reveal ? cycle_t() - k * REVEAL_STEP : HILITE_DUR;
            //   draw_hilite(r, px, my, CMW, CMH,
            //               std::clamp(kage / HILITE_DUR, 0.0, 1.0),
            //               mode_hue(k));

            if (k > 0)
                glyph(r, px - 0.5 * CMGAP, my + 0.5 * CMH, "+",
                      Rendering::palette::text_dim(), 22);
        }
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            initialize();
        if (r->is_key_pressed(Rendering::keys::Up))
            m_rank = std::min(m_rank + 1, N_MODES);
        if (r->is_key_pressed(Rendering::keys::Down))
            m_rank = std::max(m_rank - 1, 1);
        if (r->is_key_pressed(Rendering::keys::M))
            m_modal_mix = !m_modal_mix;
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

    // one hue per FAMILY, reused everywhere that family appears: both panels of
    // a pair, both KE bars, and its share of the modal mix. giving the two
    // halves of a quadrature pair separate colours implied a distinction that
    // isn't physical
    static Rendering::Color family_hue(int f) {
        switch (f % 4) {
        case 0:
            return Rendering::palette::accent4(); // blue
        case 1:
            return Rendering::palette::accent1(); // red
        case 2:
            return Rendering::palette::accent2();
        default:
            return Rendering::palette::accent3();
        }
    }

    static Rendering::Color mode_hue(int k) { return family_hue(k / PAIR); }

    // sign rides on opacity rather than colour, so the panel keeps the shared
    // speed ramp and still shows which way each lobe is currently blowing: the
    // lobes a_k(t) drives positive sit solid, the opposite-signed ones fall
    // back toward MODE_FLOOR, and the two swap as a_k crosses zero. modulating
    // the whole panel by |a_k| instead read as a global pulse and lost the
    // alternation entirely
    static double sign_alpha(double amp, double sh) {
        const double s = std::clamp(SIGN_GAIN * amp * sh, -1.0, 1.0);
        return MODE_FLOOR + (1.0 - MODE_FLOOR) * (0.5 + 0.5 * s);
    }

    static void init_field(Rendering::FieldView &fv, bool bar) {
        fv.init(COLS, ROWS,
                {.supersample = SS,
                 .edge_fade_px = FADE_PX,
                 .gamma = FIELD_GAMMA,
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

    // the streamwise component alternates sign between successive vortices, so
    // it is what carries the sign; scaled separately from the speed peak so the
    // alternation uses its full swing
    static double mode_peak_u(const VectorXd &m) {
        const int nc = (int)m.size() / 2;
        double peak = 1e-12;
        for (int c = 0; c < nc; c++)
            peak = std::max(peak, std::abs(m[c]));
        return peak;
    }

    // applied to the energy ratio before `grow`, so the reveal still eases the
    // bar up from zero rather than snapping to the floor
    static double bar_frac(double f) {
        return BAR_MIN +
               (1.0 - BAR_MIN) * std::pow(std::clamp(f, 0.0, 1.0), BAR_GAMMA);
    }

    static double ease_out(double x) {
        x = std::clamp(x, 0.0, 1.0);
        const double u = 1.0 - x;
        return 1.0 - u * u * u;
    }

    // ---- modal mix ----
    //
    // hue says which family owns a cell, luminance stays on the reconstructed
    // speed. weight is ||a_f(t)|| * ||phi_f(c)||: the second factor is fixed
    // geometry and the first is a scalar, so the colour territories hold still
    // while vortices convect through them. tying hue to a_k(t)*phi_k directly
    // would have strobed the whole panel at the shedding frequency.
    //
    // both axes are baked into a LUT because the alternative is ~10 pow() per
    // texel over an 400x200 texture. g is the leader's share, s the speed
    static constexpr int MIX_G = 32, MIX_S = 64;

    void prime_mix() {
        m_mix_n = std::min(m_num_fam, (rank() + PAIR - 1) / PAIR);
        if (m_mix_n <= 0)
            return;

        const Rendering::Oklab bg =
            Rendering::to_oklab(Rendering::palette::background());
        const Rendering::Oklab white =
            Rendering::to_oklab(Rendering::Color::hex(0xFFFFFFFFu));
        const Rendering::Oklab lead = Rendering::to_oklab(family_hue(0));

        // families past the first collapse into one hue so the LUT stays 2-D.
        // exact for the four modes the reel shows, an approximation only in the
        // wide view where the rank goes past 4
        Rendering::Oklab rest = lead;
        double rw = 0.0;
        for (int f = 1; f < m_mix_n; f++)
            rw += m_fam_amp[f];
        if (m_mix_n > 1) {
            rest = {};
            for (int f = 1; f < m_mix_n; f++) {
                const double g =
                    rw > 1e-12 ? m_fam_amp[f] / rw : 1.0 / (m_mix_n - 1);
                const Rendering::Oklab h = Rendering::to_oklab(family_hue(f));
                rest.L += g * h.L;
                rest.a += g * h.a;
                rest.b += g * h.b;
            }
        }

        for (int gi = 0; gi < MIX_G; gi++) {
            // pow-then-normalise on the two shares, done here rather than per
            // texel: identical result, and the envelope it acts on is smooth,
            // so this steepens the boundary without ever breaking it up
            const double g = (double)gi / (MIX_G - 1);
            const double p = std::pow(g, MIX_SHARP);
            const double q = std::pow(1.0 - g, MIX_SHARP);
            const double gs = p + q > 1e-30 ? p / (p + q) : 0.5;

            const Rendering::Oklab lo = Rendering::oklab_lerp(rest, lead, gs);
            const Rendering::Oklab hi =
                Rendering::oklab_lerp(lo, white, MIX_WHITE);
            for (int si = 0; si < MIX_S; si++) {
                const double s =
                    std::pow((double)si / (MIX_S - 1), FIELD_GAMMA);
                m_mix_lut[gi * MIX_S + si] = Rendering::from_oklab(
                    s < MIX_KNEE
                        ? Rendering::oklab_lerp(bg, lo, s / MIX_KNEE)
                        : Rendering::oklab_lerp(
                              lo, hi, (s - MIX_KNEE) / (1.0 - MIX_KNEE)));
            }
        }
    }

    Rendering::Color mix_color(int c, double speed01) const {
        if (m_mix_n <= 0)
            return Rendering::palette::background();

        const double e0 = m_fam_amp[0] * m_fam_env[0][c];
        double er = 0.0;
        for (int f = 1; f < m_mix_n; f++)
            er += m_fam_amp[f] * m_fam_env[f][c];

        const double tot = e0 + er;
        const double g = tot > 1e-30 ? e0 / tot : 1.0;
        const int gi = (int)(std::clamp(g, 0.0, 1.0) * (MIX_G - 1) + 0.5);
        const int si = (int)(std::clamp(speed01, 0.0, 1.0) * (MIX_S - 1) + 0.5);
        return m_mix_lut[gi * MIX_S + si];
    }

    int rank() const { return std::clamp(m_rank, 1, m_pod.num_modes()); }

    static constexpr double CYCLE = CELL_MODES * REVEAL_STEP + REVEAL_HOLD;
    double cycle_t() const { return std::fmod(m_reveal_t, CYCLE); }
    int revealed() const {
        return std::min(CELL_MODES, 1 + (int)(cycle_t() / REVEAL_STEP));
    }
    double newest_age() const {
        return cycle_t() - (revealed() - 1) * REVEAL_STEP;
    }

    // A recompute re-solves the SVD on a shifted snapshot window. Inside a
    // degenerate pair the singular values are equal to within noise, so the
    // basis it returns is only defined up to a rotation of that 2-D subspace --
    // and for a lone mode, up to a sign. The reconstruction is invariant to
    // both, which is why the summed field looks fine, but each mode panel jumps
    // to a different phase and a rank-1 reconstruction visibly leaps.
    //
    // Orthogonal Procrustes per family puts the new basis back onto the one
    // already on screen: minimise ||B Q - A|| over orthogonal Q, whose solution
    // is U V^T from the SVD of B^T A. Q is 2x2, so this costs nothing.
    void align_to_previous(std::array<VectorXd, N_MODES> &next, int n) const {
        for (int k0 = 0; k0 < n; k0 += PAIR) {
            const int m = std::min(PAIR, n - k0);
            if (m_mode_data[k0].size() != next[k0].size())
                return; // first solve, nothing to align against

            MatrixXd C(m, m);
            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++)
                    C(a, b) = next[k0 + a].dot(m_mode_data[k0 + b]);

            const Eigen::JacobiSVD<MatrixXd> s(C, Eigen::ComputeFullU |
                                                      Eigen::ComputeFullV);
            const MatrixXd Q = s.matrixU() * s.matrixV().transpose();

            std::array<VectorXd, PAIR> rot;
            for (int b = 0; b < m; b++) {
                rot[b] = Q(0, b) * next[k0];
                for (int a = 1; a < m; a++)
                    rot[b] += Q(a, b) * next[k0 + a];
            }
            for (int b = 0; b < m; b++)
                next[k0 + b] = std::move(rot[b]);
        }
    }

    void cache_modes() {
        m_num_shown = std::min(N_MODES, m_pod.num_modes());

        std::array<VectorXd, N_MODES> next;
        for (int k = 0; k < m_num_shown; k++)
            next[k] = m_pod.mode(k);
        if (m_have_pod)
            align_to_previous(next, m_num_shown);

        for (int k = 0; k < m_num_shown; k++) {
            m_mode_data[k] = std::move(next[k]);
            m_mode_scale[k] = mode_peak_speed(m_mode_data[k]);
            m_sign_scale[k] = mode_peak_u(m_mode_data[k]);
            // sigma*V^T for this mode: its RMS over the snapshot window is the
            // amplitude the live coefficient is measured against
            const VectorXd a = m_pod.coeffs(k);
            m_coef_rms[k] =
                a.size()
                    ? std::max(1e-12, a.norm() / std::sqrt((double)a.size()))
                    : 1.0;
        }
        cache_envelopes();
    }

    // ||phi_f(c)||, over both velocity components and both members of the pair.
    // for a quadrature pair this is the envelope of the wave train rather than
    // the wave, so it has no zero crossings and no stripes -- which is what
    // makes it safe to sharpen later. fixed geometry, so it is cached, not
    // recomputed per frame
    void cache_envelopes() {
        constexpr int nc = COLS * ROWS;
        m_num_fam = (m_num_shown + PAIR - 1) / PAIR;
        for (int f = 0; f < m_num_fam; f++) {
            ArrayXd s = ArrayXd::Zero(nc);
            const int k1 = std::min((f + 1) * PAIR, m_num_shown);
            for (int k = f * PAIR; k < k1; k++) {
                const VectorXd &p = m_mode_data[k];
                s += p.head(nc).array().square() + p.tail(nc).array().square();
            }
            m_fam_env[f] = s.sqrt().matrix();
        }
    }

    void update_reconstruction(double dt) {
        const VectorXd x = m_recorder.sample_state(m_fluid);
        const VectorXd fluct = x - m_pod.mean();

        // summed from the aligned modes rather than POD::reconstruct, so the
        // basis being drawn and the basis being summed are the same one at
        // every rank. projecting onto the full retained subspace would be
        // invariant to the alignment, but a rank below the family boundary is
        // not, and that is exactly the case the reveal walks through
        const int r = std::min(rank(), m_num_shown);
        m_recon = m_pod.mean();

        // a_k = <phi_k, x - mean>: the live value of what the SVD stored as
        // sigma*V^T, so the mode panels can show a_k(t)*phi_k rather than a
        // frozen |phi_k|.
        //
        // a recompute re-picks the basis inside a degenerate pair, which
        // rotates its phase and steps a_k discontinuously even though the
        // flow is continuous; the filter eases that over a fraction of a shed
        // period instead of letting it snap
        const double lam = m_amp_primed ? 1.0 - std::exp(-dt / AMP_TAU) : 1.0;
        for (int k = 0; k < m_num_shown; k++) {
            m_coef[k] = m_mode_data[k].dot(fluct);
            if (k < r)
                m_recon += m_coef[k] * m_mode_data[k];
            const double target = m_coef[k] / m_coef_rms[k];
            m_amp[k] += lam * (target - m_amp[k]);
        }

        // ||a_f||, the raw (not rms-normalised) coefficient norm over the pair,
        // so the mix carries the real energy split between families. unlike a
        // single a_k this is invariant to the basis rotation a recompute picks
        // inside a degenerate pair, so it does not step
        for (int f = 0; f < m_num_fam; f++) {
            double s2 = 0.0;
            const int k1 = std::min((f + 1) * PAIR, m_num_shown);
            for (int k = f * PAIR; k < k1; k++)
                s2 += m_coef[k] * m_coef[k];
            m_fam_amp[f] += lam * (std::sqrt(s2) - m_fam_amp[f]);
        }
        m_amp_primed = true;

        // the energy split only moves when a solve swaps in, and it moves as a
        // step. eased on its own, slower constant so the bars and the KE
        // percentages slide onto the new split rather than snapping between two
        // frames. primed on the first solve so a bar's first appearance is the
        // reveal's grow easing it up, not this one racing it
        const double blam = m_bar_primed ? 1.0 - std::exp(-dt / BAR_TAU) : 1.0;
        for (int k = 0; k < m_num_shown; k++)
            m_bar[k] += blam * (m_pod.energy(k) - m_bar[k]);
        m_bar_primed = true;

        const double area = CELL * CELL;
        const double ke_fluct = 0.5 * fluct.squaredNorm() * area;
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

        if (m_modal_mix)
            prime_mix();

        auto cell_at = [this, rox, roy](double wx, double wy) {
            const int i = (int)((wx - rox) / CELL);
            const int j = (int)((wy - roy) / CELL);
            return (i < 0 || i >= COLS || j < 0 || j >= ROWS) ? -1
                                                              : i + j * COLS;
        };

        if (m_modal_mix)
            m_recon_field.render(
                r, rox, roy, CELL,
                Rendering::FieldView::ColorSample(
                    [&, vmax](double wx, double wy, Rendering::Color &col,
                              double &a) {
                        const int c = cell_at(wx, wy);
                        if (c < 0) {
                            a = 0.0;
                            return;
                        }
                        const double u = m_recon[c], v = m_recon[nc + c];
                        col = mix_color(c, std::hypot(u, v) / vmax);
                        a = freestream_alpha(u, v);
                    }));
        else
            m_recon_field.render(
                r, rox, roy, CELL,
                Rendering::FieldView::Sample(
                    [&, vmax](double wx, double wy, double &val, double &a) {
                        const int c = cell_at(wx, wy);
                        if (c < 0) {
                            a = 0.0;
                            return;
                        }
                        const double u = m_recon[c], v = m_recon[nc + c];
                        val = std::hypot(u, v) / vmax;
                        a = freestream_alpha(u, v);
                    }));

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

        for (int k = 0; k < m_num_shown; k++) {
            const int col = k % MODE_COLS, row = k / MODE_COLS;
            const double pox = x0 + col * (MW + MGAP);
            const double poy = top_edge - MH - row * rowstep; // mode 0 top
            const double scale = m_mode_scale[k];

            const double amp = m_amp[k];
            const double us = m_sign_scale[k];
            constexpr int nc = COLS * ROWS;
            m_mode_views[k].render(
                r, pox, poy, MCELL,
                [this, k, pox, poy, scale, amp, us](double wx, double wy,
                                                    double &val, double &a) {
                    const int i = (int)((wx - pox) / MCELL);
                    const int j = (int)((wy - poy) / MCELL);
                    if (i < 0 || i >= COLS || j < 0 || j >= ROWS) {
                        a = 0.0;
                        return;
                    }
                    const VectorXd &m = m_mode_data[k];
                    const int c = i + j * COLS;
                    val = std::hypot(m[c], m[nc + c]) / scale;
                    a = std::clamp(val * sign_alpha(amp, m[c] / us), 0.0, 1.0);
                });

            const Vector2d mc = Vector2d(pox, poy) + (m_center - o) * sc;
            r->draw_circle(mc.x(), mc.y(), RADIUS * sc,
                           Rendering::palette::foreground());

            label(r, pox, poy + MH, sfmt("MODE %d", k),
                  Rendering::palette::text());
            double cum = 0.0;
            for (int i = 0; i <= k; i++)
                cum += m_bar[i];
            label(r, pox, poy + MH,
                  sfmt("E %.1f%%  cum %.0f%%", 100.0 * m_bar[k], 100.0 * cum),
                  Rendering::palette::text_dim(), 14, 1);
        }
    }

    // one field panel, clipped to CLIP of the domain width. FieldView always
    // blits its full COLS*cell quad, so the clip is done in the sampler by
    // dropping alpha past the cut rather than by resizing the texture
    // px is where the VISIBLE window starts, so the texture is offset left by
    // the clipped run. FieldView always blits its full COLS*cell quad, so both
    // cuts are made by dropping alpha rather than by resizing the texture
    static Vector2d panel_origin(double px, double py, double cell) {
        return Vector2d(px - CLIP_L * COLS * cell, py);
    }

    static bool clip_ij(const Vector2d &fo, double cell, double wx, double wy,
                        int &i, int &j) {
        const double dx = wx - fo.x();
        if (dx < CLIP_L * COLS * cell || dx > CLIP_R * COLS * cell)
            return false;
        i = (int)(dx / cell);
        j = (int)((wy - fo.y()) / cell);
        return i >= 0 && i < COLS && j >= 0 && j < ROWS;
    }

    void
    draw_panel(Rendering::Renderer *r, Rendering::FieldView &fv, double px,
               double py, double cell,
               const std::function<void(int, int, double &, double &)> &at) {
        const Vector2d fo = panel_origin(px, py, cell);
        fv.render(
            r, fo.x(), fo.y(), cell,
            Rendering::FieldView::Sample(
                [&, fo, cell](double wx, double wy, double &val, double &a) {
                    int i, j;
                    if (!clip_ij(fo, cell, wx, wy, i, j)) {
                        a = 0.0;
                        return;
                    }
                    at(i, j, val, a);
                }));
    }

    void draw_panel_mix(
        Rendering::Renderer *r, Rendering::FieldView &fv, double px, double py,
        double cell,
        const std::function<void(int, int, Rendering::Color &, double &)> &at) {
        const Vector2d fo = panel_origin(px, py, cell);
        fv.render(r, fo.x(), fo.y(), cell,
                  Rendering::FieldView::ColorSample(
                      [&, fo, cell](double wx, double wy, Rendering::Color &col,
                                    double &a) {
                          int i, j;
                          if (!clip_ij(fo, cell, wx, wy, i, j)) {
                              a = 0.0;
                              return;
                          }
                          at(i, j, col, a);
                      }));
    }

    // Pinned to Layer::Grid, which is what puts it under the field: an unpinned
    // draw_line resolves to Layer::Content, and Content paints AFTER Field, so
    // a grid built out of raw lines lands on top of the fluid. Only
    // Renderer::draw_grid carries Grid implicitly -- hand-drawn lines have to
    // say so. LayerScope restores whatever the caller had.
    //
    // Divisions are counted rather than stepped so the spacing lands evenly and
    // no sliver cell is left against either border.
    void draw_mode_box(Rendering::Renderer *r, double px, double py, double w,
                       double h) const {
        Rendering::LayerScope grid(r, Rendering::Layer::Grid);

        const double cx = px + 0.5 * w, cy = py + 0.5 * h;
        const double b = MG_BORDER * h;

        // draw_rounded_rect fills, so the frame is a larger rect in the axis
        // colour with the panel painted back over it -- the same outline trick
        // draw_ke_bar uses
        r->draw_rounded_rect(cx, cy, w + 2.0 * b, h + 2.0 * b,
                             Rendering::palette::grid_axis(), 0.0, MG_ROUND);
        r->draw_rounded_rect(cx, cy, w, h, Rendering::palette::background(),
                             0.0, MG_ROUND);

        const int nx = std::max(1, (int)std::lround(w / (MG_PITCH * h)));
        const int ny = std::max(1, (int)std::lround(h / (MG_PITCH * h)));

        const Rendering::Color maj = Rendering::palette::grid_line();
        const Rendering::Color min = Rendering::color_lerp(
            Rendering::palette::background(), maj, MG_MIN_A);

        for (int i = 1; i < nx * MG_SUB; i++) {
            const bool major = i % MG_SUB == 0;
            const double x = px + w * (double)i / (nx * MG_SUB);
            r->draw_line(x, py, x, py + h, major ? MG_TH_MAJ : MG_TH_MIN,
                         major ? maj : min);
        }
        for (int j = 1; j < ny * MG_SUB; j++) {
            const bool major = j % MG_SUB == 0;
            const double y = py + h * (double)j / (ny * MG_SUB);
            r->draw_line(px, y, px + w, y, major ? MG_TH_MAJ : MG_TH_MIN,
                         major ? maj : min);
        }
    }

    // rect is the visible extent, so h is exactly what you see and the base
    // stays pinned at y0 whatever h is
    static void bar_v(Rendering::Renderer *r, double cx, double y0, double h,
                      double w, Rendering::Color c) {
        if (h <= 0.0)
            return;
        r->draw_rounded_rect(cx, y0 + 0.5 * h, w, h, c, 0.0, BAR_ROUND);
    }

    // spans the same y as the mode's hilite box, so bar and panel read as one
    void draw_ke_bar(Rendering::Renderer *r, double bx, double by, double frac,
                     int k) {
        const double cx = bx + 0.5 * CBAR_W;
        const double y0 = by - HILITE_M, h = CMH + 2.0 * HILITE_M;
        const double b = CBAR_W * 0.09;
        const Rendering::Color hue = mode_hue(k);

        bar_v(r, cx, y0 - b, h + 2.0 * b, CBAR_W + 2.0 * b,
              Rendering::palette::foreground());
        bar_v(r, cx, y0, h, CBAR_W, Rendering::palette::background());

        const double f = std::clamp(frac, 0.0, 1.0);
        if (f < 1e-3)
            return;

        bar_v(r, cx, y0, f * h, CBAR_W,
              Rendering::color_lerp(Rendering::palette::background(), hue,
                                    0.25 + 0.75 * f));
    }

    // box round each mode that has joined the sum: fades in with the mode and
    // then stays, so the borders accumulate as the sum is built. the left
    // margin is tucked in tight because the "+" sits immediately to its left
    void draw_hilite(Rendering::Renderer *r, double px, double py, double w,
                     double h, double f, Rendering::Color c) const {
        c.a = (unsigned char)(255 * std::clamp(f, 0.0, 1.0));
        const double m = HILITE_M, ml = 0.015;
        const double x0 = px - ml, y0 = py - m, x1 = px + w + m,
                     y1 = py + h + m;
        r->draw_line(x0, y0, x1, y0, 2.0f, c);
        r->draw_line(x1, y0, x1, y1, 2.0f, c);
        r->draw_line(x1, y1, x0, y1, 2.0f, c);
        r->draw_line(x0, y1, x0, y0, 2.0f, c);
    }

    // screen-space glyph centred on a world point, for the + / = that carry
    // the summation. world_to_screen so it tracks whatever slot scales the cell
    void glyph(Rendering::Renderer *r, double wx, double wy,
               const std::string &g, Rendering::Color c, int sz) {
        Rendering::LayerScope txt(r, Rendering::Layer::Text);
        int sx, sy;
        r->world_to_screen(wx, wy, &sx, &sy);
        r->draw_text(g, sx - r->measure_text(g, sz) / 2, sy - sz / 2, sz, c);
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
        hud.small_text("[Up/Down] rank  [M] modal mix  [R] reset  [H] home",
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
    std::array<double, N_MODES> m_mode_scale{}; // peak |phi|, sets the colour
    std::array<double, N_MODES> m_sign_scale{}; // peak |phi_u|, sets the sign
    std::array<double, N_MODES> m_coef{};       // a_k for the current snapshot
    std::array<double, N_MODES> m_coef_rms{};   // its RMS over the window
    std::array<double, N_MODES> m_amp{};        // a_k/rms, temporally filtered
    std::array<double, N_MODES> m_bar{};        // displayed energy, eased
    bool m_amp_primed = false;
    bool m_bar_primed = false;
    VectorXd m_recon;

    std::array<VectorXd, N_FAM> m_fam_env; // ||phi_f(c)||, cached geometry
    std::array<double, N_FAM> m_fam_amp{}; // ||a_f(t)||, filtered
    std::array<Rendering::Color, MIX_G * MIX_S> m_mix_lut{};
    int m_num_fam = 0;
    int m_mix_n = 0;
    bool m_modal_mix = false;

    std::future<AI::POD> m_pod_job;
    AI::POD m_pending; // solved, waiting for a reveal boundary to swap in
    bool m_have_pending = false;
    bool m_job_running = false;
    int m_reveal_step = -1;
    bool m_reveal_edge = false;

    bool m_have_pod = false;
    int m_num_shown = 0;
    int m_rank = 4;
    bool m_auto_reveal = false;
    double m_reveal_t = 0.0;
    double m_ke_rel = 0.0;
    double m_time = 0.0;
    int m_frame = 0;
};

} // namespace manifold::Demo
