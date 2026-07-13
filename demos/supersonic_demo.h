#pragma once

#include <manifold/compressible/euler_2d.h>
#include <manifold/renderer/demo_base.h>
#include <manifold/renderer/field_view.h>

#include <Eigen/Dense>

#include "raylib.h"

#include <algorithm>
#include <cmath>

namespace manifold::Demo {

using Vector2d = Eigen::Vector2d;
namespace C = manifold::Compressible;

// live 2D compressible Euler: a supersonic stream forms an oblique shock off a
// wedge. rendered as synthetic schlieren (|grad rho|) so the shock glows.
class SupersonicDemo : public DemoBase {
  public:
    static constexpr int NX = 300;
    static constexpr int NY = 150;
    static constexpr double CELL = 0.02; // world units / cell (Lx=6, Ly=3)
    static constexpr int SS = 1;
    static constexpr int STEPS_PER_FRAME = 8;

    static constexpr double WEDGE_DEG = 15.0;
    static constexpr double X_TIP = 1.8;     // local x of the wedge tip
    static constexpr double SCHLIEREN = 1.2; // |grad rho| -> [0,1] scale

    const char *name() const override { return "Supersonic Wedge"; }

    void initialize() override {
        m_mach = 2.5;
        reset_flow();
        m_field.init(NX, NY,
                     {.supersample = SS, .gamma = 0.8, .colorbar = false},
                     schlieren_ramp());
    }

    void process(double) override {
        // sim time is decoupled from wall time; march a fixed budget per frame
        for (int k = 0; k < STEPS_PER_FRAME; k++)
            m_euler.step(m_euler.cfl_dt(0.4));
    }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = origin();

        m_field.render(r, o.x(), o.y(), CELL,
                       [this, o](double wx, double wy, double &val, double &a) {
                           int i, j;
                           if (!cell_at(wx, wy, o, &i, &j)) {
                               a = 0.0;
                               return;
                           }
                           val = std::clamp(m_euler.schlieren(i, j) / SCHLIEREN,
                                            0.0, 1.0);
                           a = 1.0;
                       });

        draw_wedge(r, o);
        draw_shock_line(r, o);

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("SUPERSONIC WEDGE", Rendering::palette::accent2());
        hud.small_text("2D compressible Euler . HLL flux . live",
                       Rendering::palette::text());
        hud.line(Rendering::palette::text(), "Mach: %.2f", m_mach);
        hud.line(Rendering::palette::accent3(), "wedge: %.0f deg   shock: %s",
                 WEDGE_DEG, beta_label());
        hud.separator();
        hud.small_text("[Up/Down] Mach   [R] reset",
                       Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            reset_flow();
        if (r->is_key_pressed(Rendering::keys::Up)) {
            m_mach = std::min(m_mach + 0.25, 5.0);
            reset_flow();
        }
        if (r->is_key_pressed(Rendering::keys::Down)) {
            m_mach = std::max(m_mach - 0.25, 1.5);
            reset_flow();
        }
    }

  private:
    Vector2d origin() const {
        return Vector2d(-NX * CELL * 0.5, -NY * CELL * 0.5);
    }

    void reset_flow() {
        m_euler.init_inflow(m_mach);
        m_euler.set_wedge(X_TIP, WEDGE_DEG * M_PI / 180.0);
    }

    bool cell_at(double wx, double wy, const Vector2d &o, int *i,
                 int *j) const {
        const int ci = (int)((wx - o.x()) / CELL);
        const int cj = (int)((wy - o.y()) / CELL);
        if (ci < 0 || ci >= NX || cj < 0 || cj >= NY)
            return false;
        *i = ci;
        *j = cj;
        return m_euler.solid(ci, cj) ? false : true; // wedge drawn separately
    }

    // the wedge triangle, filled (screen space) + outline
    void draw_wedge(Rendering::Renderer *r, const Vector2d &o) const {
        const double t = std::tan(WEDGE_DEG * M_PI / 180.0);
        const double Lx = NX * CELL;
        const Vector2d tip = o + Vector2d(X_TIP, 0.0);
        const Vector2d br = o + Vector2d(Lx, 0.0);
        const Vector2d tr = o + Vector2d(Lx, (Lx - X_TIP) * t);
        int x0, y0, x1, y1, x2, y2;
        r->world_to_screen(tip.x(), tip.y(), &x0, &y0);
        r->world_to_screen(br.x(), br.y(), &x1, &y1);
        r->world_to_screen(tr.x(), tr.y(), &x2, &y2);
        const Rendering::Color fg = Rendering::palette::foreground();
        const Color fill{(unsigned char)(fg.r / 3), (unsigned char)(fg.g / 3),
                         (unsigned char)(fg.b / 3), 255};
        DrawTriangle({(float)x0, (float)y0}, {(float)x1, (float)y1},
                     {(float)x2, (float)y2}, fill);
        DrawTriangle({(float)x0, (float)y0}, {(float)x2, (float)y2},
                     {(float)x1, (float)y1}, fill);
        r->draw_line(tip.x(), tip.y(), tr.x(), tr.y(), 2.0f, fg);
    }

    // overlay the analytic weak-shock ray from the tip (theta-beta-M relation)
    void draw_shock_line(Rendering::Renderer *r, const Vector2d &o) const {
        const double beta = oblique_beta(m_mach, WEDGE_DEG * M_PI / 180.0);
        if (beta < 0.0)
            return; // detached -> no attached ray to draw
        const Vector2d tip = o + Vector2d(X_TIP, 0.0);
        const double len = NY * CELL;
        const Vector2d end =
            tip + len * Vector2d(std::cos(beta), std::sin(beta));
        r->draw_line(tip.x(), tip.y(), end.x(), end.y(), 1.5f,
                     Rendering::palette::accent4());
    }

    const char *beta_label() const {
        const double beta = oblique_beta(m_mach, WEDGE_DEG * M_PI / 180.0);
        static char buf[24];
        if (beta < 0.0)
            std::snprintf(buf, sizeof(buf), "detached");
        else
            std::snprintf(buf, sizeof(buf), "%.0f deg", beta * 180.0 / M_PI);
        return buf;
    }

    // weak-shock root of the theta-beta-M relation (scan up from the Mach
    // angle)
    static double oblique_beta(double M, double theta) {
        const double mu = std::asin(1.0 / M);
        for (double b = mu + 1e-3; b < M_PI / 2; b += 5e-4) {
            const double s = std::sin(b);
            const double th =
                std::atan(2.0 / std::tan(b) * (M * M * s * s - 1.0) /
                          (M * M * (C::GAMMA + std::cos(2 * b)) + 2.0));
            if (th >= theta)
                return b;
        }
        return -1.0;
    }

    // black -> red -> orange -> white "fire" ramp for the schlieren
    static Rendering::Colormap schlieren_ramp() {
        return [](double t) -> Rendering::Color {
            t = std::clamp(t, 0.0, 1.0);
            auto mix = [](Rendering::Color a, Rendering::Color b, double f) {
                return Rendering::Color::rgba(
                    (unsigned char)(a.r + f * (b.r - a.r)),
                    (unsigned char)(a.g + f * (b.g - a.g)),
                    (unsigned char)(a.b + f * (b.b - a.b)), 255);
            };
            const auto c0 = Rendering::Color::rgba(8, 10, 18, 255);
            const auto c1 = Rendering::Color::rgba(120, 20, 30, 255);
            const auto c2 = Rendering::Color::rgba(240, 120, 30, 255);
            const auto c3 = Rendering::Color::rgba(255, 245, 220, 255);
            if (t < 0.4)
                return mix(c0, c1, t / 0.4);
            if (t < 0.75)
                return mix(c1, c2, (t - 0.4) / 0.35);
            return mix(c2, c3, (t - 0.75) / 0.25);
        };
    }

    C::Euler2D m_euler{NX, NY, CELL, CELL};
    double m_mach = 2.5;
    Rendering::FieldView m_field;
};

} // namespace manifold::Demo
