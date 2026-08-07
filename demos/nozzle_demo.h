#pragma once

#include <manifold/compressible/euler_2d.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>
#include <manifold/renderer/interpolation.h>
#include <manifold/renderer/texture_view.h>

#include <Eigen/Dense>

#include "manifold/renderer/theme.h"
#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
namespace C = manifold::Compressible;

// axisymmetric under-expanded jet from a bell nozzle. the half-plane Euler sim
// (axis = y, r >= 0) is mirrored about the axis so you see the full round jet:
// a shock-diamond train that ends in a Mach disk. adjust ambient / chamber
// pressure to move between over- and under-expanded.
class NozzleDemo : public DemoBase {
  public:
    // grid refinement; NX*CELL and NY*CELL stay at 4.320 x 1.008 of world, so
    // the field keeps its extent and the slot fitted to it is untouched
    static constexpr double RES_SCALE = 1.0;
    static constexpr int NX = (int)(360 * RES_SCALE);
    static constexpr int NY = (int)(84 * RES_SCALE);
    static constexpr double CELL = 0.012 / RES_SCALE;
    // The plume advances at STEPS_PER_FRAME * cfl_dt of sim time per frame.
    // Buying that with more steps costs linearly; buying it with a larger CFL
    // number is nearly free. Going 8 -> 26 steps was what made this lag.
    // cfl_dt scales with CELL, so the step count has to rise with RES_SCALE or
    // a refined grid silently plays the plume back slower. Cost is cubic.
    static constexpr int STEPS_PER_FRAME = (int)(10 * RES_SCALE);
    static constexpr double CFL = 0.60;

    static constexpr double Me = 2.2;    // nozzle exit Mach
    static constexpr double RHO_E = 1.6; // exit density
    // denominated in cells, so it scales or the throat physically shrinks
    static constexpr int EXIT_R = (int)(8 * RES_SCALE);
    static constexpr double SCHLIEREN = 0.9;

    // F-22 nozzle, rasterised from the source SVG and measured off the asset.
    // The lower body is filled with the background colour, so it OCCLUDES --
    // which puts the exit plane at the trailing edge rather than up inside the
    // nozzle. Anchoring it higher would bury the first 0.26 world units of
    // plume behind the body, and that is where the exit shock structure sits:
    // the densest, most legible part of the schlieren.
    //
    // Dimensions come from the loaded texture, never from constants -- a
    // hardcoded size silently shears the blit the moment the asset is re-cut.
    // this drawing is symmetric, so the centreline is the image centre; the
    // body runs the full image width and the V starts converging at row 331,
    // which is where the exit plane sits so the chevron overhangs the plume
    // rasterised with the viewBox padded 3 units, because the source frame is
    // tight to the artwork and the outermost strokes were being clipped by it
    // on export -- one side lost its outer half, the other kept it
    static constexpr double ART_AXIS = 109.2; // px, nozzle centreline
    static constexpr double ART_EXIT = 343.0; // px from the top, exit plane
    static constexpr double ART_HALF = 96.8;  // px, nozzle body half-width
    // 1.0 makes the nozzle interior exactly as wide as the jet. The jet is
    // fixed at EXIT_R cells, so anything above 1.0 buys a bigger nozzle by
    // making it wider than its own exhaust -- which reads as an over-expanded
    // jet contracting after the lip, not as an error, but it is a real trade.
    // Raising EXIT_R instead would grow both together and keep them flush.
    static constexpr double ART_GAIN = 2.0;
    static constexpr double ART_S =
        ART_GAIN * (EXIT_R * CELL) / ART_HALF; // world units per px

    // caption anchor, in SOLVER coords, so to_draw() carries it through the
    // quarter turn with everything else. x is downstream of the exit, y is out
    // past the jet's own half-width
    //
    // cap_x = -0.6 is nice top right
    static constexpr double CAP_X = -0.6, CAP_Y = 0.50;
    static constexpr int CAP_SIZE = 13;

    // Procedural art instead of the PNG. Every path in the source SVG is
    // straight-line only, so the transcription is exact rather than an
    // approximation. Worth it because it removes the whole failure mode we hit
    // with the texture: no minification, so no thin-stroke aliasing, no
    // mipmaps, no sub-pixel phase making one outline heavier than the other.
    // Also resolution-independent, so ART_GAIN can go anywhere.
    static constexpr bool ART_PROCEDURAL = true;

    // Art vertices are in SVG units, origin on the nozzle AXIS at the exit
    // plane, +y upstream. Using the body's own axis rather than the viewBox
    // centre quietly corrects the 0.354-unit asymmetry still in the drawing.
    static constexpr double ART_HALF_SVG = 27.1983; // body half-width
    static constexpr double ART_S_SVG =
        ART_GAIN * (EXIT_R * CELL) / ART_HALF_SVG; // world units per SVG unit

    const char *name() const override { return "Nozzle Plume"; }
    double default_cam_x() const override { return 1.1; }
    double default_cam_y() const override { return 0.0; }
    double default_cam_zoom() const override { return 300.0; }

    // a quarter turn clockwise, so the plume runs down the frame. done at the
    // source (swapped field dims + remapped sampling) rather than by rotating
    // the renderer, because draw_texture blits an axis-aligned quad
    void set_quarter_turn(bool on) { m_turned = on; }

    void initialize() override {
        m_pamb = 1.0;
        m_pe = 1.7;
        reset_flow();
        if (m_turned)
            m_field.init(2 * NY, NX,
                         {.supersample = 1,
                          .edge_fade_frac = 0.055,
                          .gamma = 0.8,
                          .colorbar = false},
                         fire_ramp());
        else
            m_field.init(NX, 2 * NY,
                         {.supersample = 1,
                          .edge_fade_frac = 0.055,
                          .gamma = 0.8,
                          .colorbar = false},
                         fire_ramp());
        load_art();
    }

    void process(double) override {
        for (int k = 0; k < STEPS_PER_FRAME; k++) {
            stamp_nozzle();
            m_euler.step(m_euler.cfl_dt(CFL));
        }
    }

    void render(Rendering::Renderer *r) override {
        render_cell(r);

        draw_hud(r);
    }

    void render_cell(Rendering::Renderer *r) override {
        const double R = NY * CELL;
        const double ox = m_turned ? -R : 0.0;
        const double oy = m_turned ? -NX * CELL : -R;

        // mirrored schlieren: |y| maps to the radial cell
        m_field.render(r, ox, oy, CELL,
                       [this](double wx, double wy, double &val, double &a) {
                           const Vector2d p = to_solver(wx, wy);
                           int i, j;
                           if (!cell_at(p.x(), p.y(), &i, &j)) {
                               a = 0.0;
                               return;
                           }
                           val = std::clamp(m_euler.schlieren(i, j) / SCHLIEREN,
                                            0.0, 1.0);
                           a = 1.0;
                       });

        draw_nozzle(r);
        draw_caption(r);
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            reset_flow();
        if (r->is_key_pressed(Rendering::keys::Up)) {
            m_pamb = std::min(m_pamb + 0.1, 3.0);
            m_euler.update_ambient(1.0, m_pamb);
        }
        if (r->is_key_pressed(Rendering::keys::Down)) {
            m_pamb = std::max(m_pamb - 0.1, 0.2);
            m_euler.update_ambient(1.0, m_pamb);
        }
        if (r->is_key_pressed(Rendering::keys::Right))
            m_pe = std::min(m_pe + 0.1, 4.0);
        if (r->is_key_pressed(Rendering::keys::Left))
            m_pe = std::max(m_pe - 0.1, 0.3);
    }

  private:
    double exit_speed() const {
        return Me * std::sqrt(C::GAMMA_EXHAUST * m_pe / RHO_E);
    }

    void reset_flow() {
        m_euler.init_ambient(1.0, m_pamb);
        m_euler.set_axisymmetric(true);
        m_euler.set_bc(Euler2D_BC::Farfield, Euler2D_BC::Farfield,
                       Euler2D_BC::Wall, Euler2D_BC::Farfield);
        stamp_nozzle();
    }

    void stamp_nozzle() {
        m_euler.clear_reservoirs();
        const auto s =
            C::Euler2D::make_state(RHO_E, exit_speed(), 0.0, m_pe, 1.0);
        for (int j = 0; j <= EXIT_R; j++)
            for (int i = 0; i < 3; i++)
                m_euler.add_reservoir(i, j, s);
    }

    // world -> radial cell, mirrored about the axis at y = 0
    bool cell_at(double wx, double wy, int *i, int *j) const {
        const int ci = (int)(wx / CELL);
        const int cj = (int)(std::abs(wy) / CELL);
        if (ci < 0 || ci >= NX || cj < 0 || cj >= NY)
            return false;
        *i = ci;
        *j = cj;
        return true;
    }

    // a filled C-D bell, mirrored, exit lip at x = 0
    // solver space has the jet along +x; drawn space turns that to -y
    Vector2d to_draw(double x, double y) const {
        return m_turned ? Vector2d(y, -x) : Vector2d(x, y);
    }
    Vector2d to_solver(double wx, double wy) const {
        return m_turned ? Vector2d(-wy, wx) : Vector2d(wx, wy);
    }

    // Demos are launched from build/, so assets resolve through ../ (the same
    // way string_art and the ESN weights do). Both spellings are tried because
    // a missing asset here is silent otherwise -- the nozzle simply does not
    // appear, with nothing to say why.
    //
    // The SVG is drawn bell-down, which is the quarter-turned orientation the
    // reel uses, so that case blits as-is and only the upright demo pays for a
    // rotation. The rotation goes through init() rather than set_pixels()
    // because the texture's own width/height have to swap with it; set_pixels
    // keeps the old dims and would upload the buffer sheared.
    void load_art() {
        // mipmapped: the asset is minified ~4x, and the outline strokes are
        // about a pixel wide on screen. bilinear renders those at 1 or 2 px
        // depending on sub-pixel phase, which is why the two sides of the
        // nozzle came out visibly different weights
        const Rendering::TextureViewSettings s{.bilinear = true,
                                               .mipmaps = true,
                                               .layer =
                                                   Rendering::Layer::Content};
        for (const char *p :
             {"../assets/images/nozzle.png", "assets/images/nozzle.png"})
            if (m_art.load(p, s))
                break;
        if (!m_art.valid())
            return;

        if (!m_turned) {
            const int w = m_art.width(), h = m_art.height();
            m_art.init(h, w, rotate_ccw(m_art.pixels(), w, h), s);
        }
        m_art.set_scale(ART_S, ART_S);
    }

    // names what the field actually is, since a schlieren image is a density
    // GRADIENT and reads nothing like the speed ramps elsewhere in the reel.
    // anchored in world space beside the exit so it tracks whatever scale the
    // slot fits the cell at, then drawn screen-space so it stays legible
    void draw_caption(Rendering::Renderer *r) const {
        const Vector2d p = to_draw(CAP_X, CAP_Y);
        Rendering::LayerScope ui(r, Rendering::Layer::Text);
        int sx, sy;
        r->world_to_screen(p.x(), p.y(), &sx, &sy);
        r->draw_text("SCHLIEREN", sx, sy, CAP_SIZE,
                     Rendering::palette::text_dim());
    }

    // 90 deg CCW, so the bell moves from the bottom edge to the right and the
    // jet runs along +x as the untuned solver expects
    static std::vector<::Color> rotate_ccw(const std::vector<::Color> &src,
                                           int w, int h) {
        std::vector<::Color> out((size_t)w * h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                out[(size_t)(w - 1 - x) * h + y] = src[(size_t)y * w + x];
        return out;
    }

    // bottom-left of the blit, putting the exit plane on the jet origin and the
    // drawing's centreline on the jet axis. the art's own centre and the
    // nozzle's centreline differ by 3 px -- the engine housing above is not
    // symmetric about the nozzle -- so this tracks the nozzle, not the image.
    // falls back to the procedural bell rather than drawing nothing, so a
    // missing asset looks like a regression instead of a plume in empty space
    // SVG-local (x across, +y upstream) -> drawn, via solver space so the
    // quarter turn is the same one everything else goes through
    Vector2d art_pt(double lx, double ly) const {
        return to_draw(-ly * ART_S_SVG, lx * ART_S_SVG);
    }

    // stroke widths are in SVG units but draw_line takes SCREEN px, so they
    // have to go through the live transform or they stop scaling with zoom
    float art_px(Rendering::Renderer *r, double svg_w) const {
        const Vector2d a = art_pt(0.0, 0.0), b = art_pt(svg_w, 0.0);
        int x0, y0, x1, y1;
        r->world_to_screen(a.x(), a.y(), &x0, &y0);
        r->world_to_screen(b.x(), b.y(), &x1, &y1);
        return (float)std::max(1, std::abs(x1 - x0));
    }

    void draw_nozzle_art(Rendering::Renderer *r) const {
#include "nozzle_art.inc"
        // the shell fill is exactly palette::background(), which is what makes
        // the body read as a cut-out and occlude the plume. a literal would
        // silently stop matching the moment the theme changed
        const Rendering::Color shell = Rendering::palette::background();
        const Rendering::Color plate = Rendering::Color::hex(0xA08C7AFF);
        const Rendering::Color ink = Rendering::Color::hex(0x000000FF);

        auto tris = [&](const float *v, int n, Rendering::Color c) {
            for (int i = 0; i + 5 < n; i += 6) {
                const Vector2d a = art_pt(v[i], v[i + 1]);
                const Vector2d b = art_pt(v[i + 2], v[i + 3]);
                const Vector2d d = art_pt(v[i + 4], v[i + 5]);
                r->draw_triangle(a.x(), a.y(), b.x(), b.y(), d.x(), d.y(), c);
            }
        };
        tris(ART_SHELL, (int)(sizeof(ART_SHELL) / sizeof(float)), shell);
        tris(ART_PLATE, (int)(sizeof(ART_PLATE) / sizeof(float)), plate);

        for (const auto &s : ART_LINE_IX) {
            const float w = art_px(r, s.w);
            for (int i = 0; i + 1 < s.n; i++) {
                const Vector2d a = art_pt(ART_LINE[2 * (s.at + i)],
                                          ART_LINE[2 * (s.at + i) + 1]);
                const Vector2d b = art_pt(ART_LINE[2 * (s.at + i + 1)],
                                          ART_LINE[2 * (s.at + i + 1) + 1]);
                r->draw_line(a.x(), a.y(), b.x(), b.y(), w, ink);
            }
        }
    }

    void draw_nozzle(Rendering::Renderer *r) {
        if (ART_PROCEDURAL) {
            // draw_nozzle_art(r);
            return;
        }
        if (!m_art.valid()) {
            draw_nozzle_procedural(r);
            return;
        }
        // once rotated the exit is a column from the left, so only the turned
        // case needs the height to convert a row into a distance off the bottom
        if (m_turned)
            m_art.render(r, -ART_AXIS * ART_S,
                         -(m_art.height() - ART_EXIT) * ART_S);
        else
            m_art.render(r, -ART_EXIT * ART_S, -ART_AXIS * ART_S);
    }

    void draw_nozzle_procedural(Rendering::Renderer *r) const {
        const double re = EXIT_R * CELL;
        const double rt = 0.25 * re, rc = 0.44 * re;
        // re is small and fixed, so a long divergent section just reads as two
        // diverging strips. the length lives in the chamber instead; the bell
        // stays short and strongly flared so it reads as a bell
        const double xe = 0.0;    // exit plane
        const double xt = -0.115; // throat
        const double xc = -0.200; // chamber front / start of convergence
        const double xh = -0.400; // injector face

        const auto fg = Rendering::palette::foreground();
        const auto bg = Rendering::palette::background();
        const auto shadow = Rendering::palette::shadow();
        const auto dim = Rendering::palette::text_dim();
        const double dth = m_turned ? -M_PI_2 : 0.0;

        auto seg = [&](double x0, double y0, double x1, double y1, float t,
                       Rendering::Color c) {
            const Vector2d a = to_draw(x0, y0), b = to_draw(x1, y1);
            r->draw_smooth_line(a.x(), a.y(), b.x(), b.y(), t, c);
        };
        auto tri = [&](double x0, double y0, double x1, double y1, double x2,
                       double y2, Rendering::Color c) {
            const Vector2d a = to_draw(x0, y0), b = to_draw(x1, y1),
                           d = to_draw(x2, y2);
            r->draw_triangle(a.x(), a.y(), b.x(), b.y(), d.x(), d.y(), c);
        };
        // wall quad spanning [ya,yb] at x0 and [yc,yd] at x1, mirrored about
        // the axis. dx/dy shift after the mirror, so the drop shadow offsets
        // in one direction instead of just fattening the profile
        auto quad = [&](double x0, double ya, double yb, double x1, double yc,
                        double yd, Rendering::Color col, double dx = 0.0,
                        double dy = 0.0) {
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                const double s = sgn;
                tri(x0 + dx, s * ya + dy, x0 + dx, s * yb + dy, x1 + dx,
                    s * yd + dy, col);
                tri(x0 + dx, s * ya + dy, x1 + dx, s * yd + dy, x1 + dx,
                    s * yc + dy, col);
            }
        };
        auto box = [&](double x0, double x1, double hh, Rendering::Color c,
                       double dx = 0.0, double dy = 0.0) {
            const Vector2d ctr = to_draw(0.5 * (x0 + x1) + dx, dy);
            r->draw_rect(ctr.x(), ctr.y(), x1 - x0, 2.0 * hh, c, dth);
        };

        auto radius_at = [&](double x) {
            if (x <= xc)
                return rc;
            if (x <= xt) {
                const double f = (x - xc) / (xt - xc);
                return rc + (rt - rc) * (0.5 - 0.5 * std::cos(M_PI * f));
            }
            const double f = (x - xt) / (xe - xt);
            return rt + (re - rt) * std::pow(f, 0.65);
        };
        // wall scales with radius so the bell stays visually heavy where it is
        // wide; the lip is a local thickening rather than a stuck-on stub
        auto wall_at = [&](double x) {
            const double f = std::clamp((x - (xe - 0.022)) / 0.022, 0.0, 1.0);
            return (0.45 * radius_at(x) + 0.020) * (1.0 + 0.28 * f * f);
        };
        const double wall = 0.45 * rc + 0.020; // chamber wall, for the head

        const int N = 40;
        // offset shadow pass first, the way the linkage bars are drawn, then
        // the body over it — that is what gives the mechanisms their weight
        for (int pass = 0; pass < 2; ++pass) {
            const double dx = pass == 0 ? 0.006 : 0.0;
            const double dy = pass == 0 ? -0.006 : 0.0;
            const auto col = pass == 0 ? shadow : fg;
            double px = xc, pr = radius_at(xc), pw = wall_at(xc);
            for (int i = 1; i <= N; ++i) {
                const double x = xc + (xe - xc) * i / N;
                const double rr = radius_at(x), ww = wall_at(x);
                quad(px, pr, pr + pw, x, rr, rr + ww, col, dx, dy);
                px = x, pr = rr, pw = ww;
            }
            // chamber and injector are solid to the axis: the solver domain
            // starts at x = 0, so there is no field to occlude back here
            quad(xh, 0.0, rc + wall * 1.20, xc + 0.014, 0.0, rc + wall, col, dx,
                 dy);
            box(xh - 0.030, xh, rc + wall * 1.55, col, dx, dy);
        }

        // at the demo's camera the whole nozzle is ~120 px, so detail has to be
        // few and bold: one stiffener, one throat collar, one chamber band
        for (double f : {0.55}) {
            const double x = xt + (xe - xt) * f, t = 0.007;
            quad(x - t, radius_at(x - t), radius_at(x - t) + wall_at(x - t),
                 x + t, radius_at(x + t), radius_at(x + t) + wall_at(x + t),
                 bg);
        }
        // joint gaps: throat collar and injector face
        quad(xt - 0.007, radius_at(xt - 0.007),
             radius_at(xt - 0.007) + wall_at(xt - 0.007), xt + 0.007,
             radius_at(xt + 0.007), radius_at(xt + 0.007) + wall_at(xt + 0.007),
             bg);
        box(xh - 0.005, xh + 0.005, rc + wall * 1.55, bg);
        box(xh + (xc - xh) * 0.55 - 0.007, xh + (xc - xh) * 0.55 + 0.007,
            rc + wall * 1.25, bg);

        // propellant feed off one side of the head
        const double fy = rc + wall * 1.20;
        seg(xh + 0.028, fy, xh + 0.028, fy + 0.048, 3.2f, fg);
        seg(xh + 0.028, fy + 0.048, xh + 0.098, fy + 0.048, 3.2f, fg);
        {
            const Vector2d c = to_draw(xh + 0.104, fy + 0.048);
            r->draw_rect(c.x(), c.y(), 0.042, 0.030, fg, dth);
        }

        // highlight along the inner wall, one side only, so it reads as lit
        double px = xc, pr = radius_at(xc);
        for (int i = 1; i <= N; ++i) {
            const double x = xc + (xe - xc) * i / N;
            const double rr = radius_at(x);
            seg(px, -pr, x, -rr, 1.2f, dim);
            px = x, pr = rr;
        }
    }

    void draw_hud(Rendering::Renderer *r) {
        const double p_amb = m_euler.ambient_pressure();
        const double thrust = m_euler.axial_thrust(6, p_amb, true);
        const double ratio = m_pe / p_amb;
        const char *regime = ratio > 1.05   ? "under-expanded"
                             : ratio < 0.95 ? "over-expanded"
                                            : "matched";
        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("NOZZLE PLUME", Rendering::palette::accent2());
        hud.small_text("axisymmetric compressible Euler . HLLC . live",
                       Rendering::palette::text());
        hud.line(Rendering::palette::text(),
                 "chamber p: %.2f   ambient p: %.2f", m_pe, p_amb);
        hud.line(Rendering::palette::accent3(), "p_e/p_amb: %.2f  (%s)", ratio,
                 regime);
        hud.line(Rendering::palette::accent1(), "thrust: %.2f", thrust);
        hud.separator();
        hud.small_text("[Up/Dn] ambient p   [L/R] chamber p   [R] reset",
                       Rendering::palette::text_dim());
    }

    static Rendering::Colormap fire_ramp() {
        return [](double t) -> Rendering::Color {
            t = std::clamp(t, 0.0, 1.0);
            auto mix = [](Rendering::Color a, Rendering::Color b, double f) {
                return Rendering::Color::rgba(
                    (unsigned char)(a.r + f * (b.r - a.r)),
                    (unsigned char)(a.g + f * (b.g - a.g)),
                    (unsigned char)(a.b + f * (b.b - a.b)), 255);
            };

            const auto c0 = Rendering::palette::background();
            const auto c1 = Rendering::Color::rgba(120, 20, 30, 255);
            const auto c2 = Rendering::Color::rgba(160, 60, 25, 255);
            const auto c3 = Rendering::Color::rgba(200, 95, 20, 255);

            if (t < 0.4)
                return mix(c0, c1, t / 0.4);
            if (t < 0.75)
                return mix(c1, c2, (t - 0.4) / 0.35);
            return mix(c2, c3, (t - 0.75) / 0.25);
        };
    }

    using Euler2D_BC = C::Euler2D::BC;
    C::Euler2D m_euler{NX, NY, CELL, CELL};
    double m_pamb = 1.0, m_pe = 1.7;
    bool m_turned = false;
    Rendering::FieldView m_field;
    Rendering::TextureView m_art;
};

} // namespace manifold::Demo
