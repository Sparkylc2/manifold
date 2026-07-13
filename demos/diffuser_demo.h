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

// supersonic diffuser, jet-inlet style: a long top ramp slants down into a
// throat (flat floor below), the flow runs supersonic with an oblique shock
// train, then a back-pressure terminal shock drops it to subsonic. the whole
// fluid is drawn opaque (coloured by Mach) so you see it ram through; dye
// streaks + shock lines glow on top.
class DiffuserDemo : public DemoBase {
  public:
    static constexpr int NX = 360;
    static constexpr int NY = 180;
    static constexpr double CELL = 0.011; // Lx ~ 3.96, Ly ~ 1.98
    static constexpr int SS = 1;
    static constexpr int STEPS_PER_FRAME = 8;

    double MACH_IN = 2.5;
    static constexpr double MACH_MAX = 3.5; // colour scale
    static constexpr double SCHLIEREN = 1.1;
    static constexpr double DYE_BOOST = 3.0; // make thin dye visible downstream
    static constexpr int BRUSH = 3;

    const char *name() const override { return "Supersonic Diffuser"; }

    void initialize() override {
        m_pback = 1.7;
        reset_flow();
        m_mach.init(NX, NY,
                    {.supersample = SS, .gamma = 1.0, .colorbar = false},
                    mach_ramp());
        m_over.init(NX, NY,
                    {.supersample = SS, .gamma = 1.0, .colorbar = false},
                    overlay_ramp());
    }

    void process(double) override {
        for (int k = 0; k < STEPS_PER_FRAME; k++) {
            const double dt = m_euler.cfl_dt(0.4);
            for (int j = NY / 2 - 30; j <= NY / 2 + 30; j += 12)
                for (int i = 2; i < 6; i++)
                    m_euler.inject_dye(i, j, 0.6 * dt * 60.0);
            m_euler.step(dt);
        }
    }

    void render(Rendering::Renderer *r) override {
        const Vector2d o = origin();

        // base layer: the whole fluid, opaque, coloured by Mach
        m_mach.render(r, o.x(), o.y(), CELL,
                      [this, o](double wx, double wy, double &val, double &a) {
                          int i, j;
                          if (!cell_at(wx, wy, o, &i, &j)) {
                              a = 0.0;
                              return;
                          }
                          val = std::clamp(m_euler.mach(i, j) / MACH_MAX, 0.0,
                                           1.0);
                          a = 1.0;
                      });

        // overlay: dye streaks + shock lines glow on top
        m_over.render(r, o.x(), o.y(), CELL,
                      [this, o](double wx, double wy, double &val, double &a) {
                          int i, j;
                          if (!cell_at(wx, wy, o, &i, &j)) {
                              a = 0.0;
                              return;
                          }
                          const double dye = std::clamp(
                              m_euler.dye_at(i, j) * DYE_BOOST, 0.0, 1.0);
                          const double sch = std::clamp(
                              m_euler.schlieren(i, j) / SCHLIEREN, 0.0, 1.0);
                          a = std::max(dye, 0.9 * sch);
                          val = (0.9 * sch >= dye) ? 1.0 : 0.0; // shock vs dye
                      });

        draw_walls(r, o);

        Rendering::HUDPanel hud(r, 12, 12);
        hud.title("SUPERSONIC DIFFUSER", Rendering::palette::accent2());
        hud.small_text("jet inlet . shock train -> subsonic diffuser",
                       Rendering::palette::text());
        hud.line(Rendering::palette::text(), "inlet Mach: %.2f", MACH_IN);
        hud.line(Rendering::palette::accent3(), "back pressure: %.2f", m_pback);
        hud.separator();
        hud.small_text("drag: dye   [L/R] Mach   [Up/Dn] back pressure   [R]",
                       Rendering::palette::text_dim());
    }

  protected:
    void on_input(Rendering::Renderer *r) override {
        if (r->is_key_pressed(Rendering::keys::R))
            reset_flow();
        if (r->is_key_pressed(Rendering::keys::Up)) {
            m_pback = std::min(m_pback + 0.2, 5.0);
            m_euler.set_back_pressure(m_pback);
        }
        if (r->is_key_pressed(Rendering::keys::Down)) {
            m_pback = std::max(m_pback - 0.2, 0.0);
            m_euler.set_back_pressure(m_pback);
        }
        if (r->is_key_pressed(Rendering::keys::Left)) {
            MACH_IN = std::max(MACH_IN - 0.2, 1.4);
            reset_flow();
        }
        if (r->is_key_pressed(Rendering::keys::Right)) {
            MACH_IN = std::min(MACH_IN + 0.2, MACH_MAX);
            reset_flow();
        }

        // paint dye with the mouse
        if (r->is_mouse_button_down(Rendering::mouse::Left)) {
            int mx, my;
            r->get_mouse_pos(&mx, &my);
            double wx, wy;
            r->screen_to_world(mx, my, &wx, &wy);
            const Vector2d o = origin();
            const int ci = (int)((wx - o.x()) / CELL);
            const int cj = (int)((wy - o.y()) / CELL);
            for (int dj = -BRUSH; dj <= BRUSH; ++dj)
                for (int di = -BRUSH; di <= BRUSH; ++di)
                    m_euler.inject_dye(ci + di, cj + dj, 0.6);
        }
    }

  private:
    Vector2d origin() const {
        return Vector2d(-NX * CELL * 0.5, -NY * CELL * 0.5);
    }

    static double lerp(double a, double b, double t) {
        t = std::clamp(t, 0.0, 1.0);
        return a + (b - a) * t;
    }

    // jet inlet: long straight top, slanting down to a throat, then a diffuser
    static double top_wall(double x) {
        const double xf = x / (NX * CELL);
        if (xf < 0.36)
            return 1.78;
        if (xf < 0.66)
            return lerp(1.78, 1.32, (xf - 0.36) / 0.30); // compression ramp
        return lerp(1.32, 1.55, (xf - 0.66) / 0.34);     // subsonic diffuser
    }
    static double bottom_wall(double) { return 0.42; } // flat floor

    void reset_flow() {
        m_euler.set_solid_mask([](double x, double y) {
            return y > top_wall(x) || y < bottom_wall(x);
        });
        m_euler.init_inflow(MACH_IN);
        m_euler.set_back_pressure(m_pback);
    }

    bool cell_at(double wx, double wy, const Vector2d &o, int *i,
                 int *j) const {
        const int ci = (int)((wx - o.x()) / CELL);
        const int cj = (int)((wy - o.y()) / CELL);
        if (ci < 0 || ci >= NX || cj < 0 || cj >= NY)
            return false;
        *i = ci;
        *j = cj;
        return !m_euler.solid(ci, cj);
    }

    void draw_walls(Rendering::Renderer *r, const Vector2d &o) const {
        const double Lx = NX * CELL;
        const auto col = Rendering::palette::foreground();
        const int seg = 90;
        for (int s = 0; s < seg; s++) {
            const double x0 = Lx * s / seg, x1 = Lx * (s + 1) / seg;
            r->draw_line(o.x() + x0, o.y() + top_wall(x0), o.x() + x1,
                         o.y() + top_wall(x1), 2.0f, col);
            r->draw_line(o.x() + x0, o.y() + bottom_wall(x0), o.x() + x1,
                         o.y() + bottom_wall(x1), 2.0f, col);
        }
    }

    // Mach ramp: navy (subsonic) -> white (sonic) -> red (supersonic)
    static Rendering::Colormap mach_ramp() {
        return [](double t) -> Rendering::Color {
            t = std::clamp(t, 0.0, 1.0);
            auto mix = [](Rendering::Color a, Rendering::Color b, double f) {
                return Rendering::Color::rgba(
                    (unsigned char)(a.r + f * (b.r - a.r)),
                    (unsigned char)(a.g + f * (b.g - a.g)),
                    (unsigned char)(a.b + f * (b.b - a.b)), 255);
            };
            const auto navy = Rendering::Color::rgba(14, 22, 52, 255);
            const auto teal = Rendering::Color::rgba(40, 140, 185, 255);
            const auto white = Rendering::Color::rgba(238, 244, 250, 255);
            const auto orange = Rendering::Color::rgba(244, 142, 44, 255);
            const auto red = Rendering::Color::rgba(196, 42, 30, 255);
            const double son = 1.0 / MACH_MAX; // sonic point on the scale
            if (t < son * 0.6)
                return mix(navy, teal, t / (son * 0.6));
            if (t < son)
                return mix(teal, white, (t - son * 0.6) / (son * 0.4));
            if (t < (1.0 + son) * 0.5)
                return mix(white, orange,
                           (t - son) / ((1.0 + son) * 0.5 - son));
            return mix(orange, red,
                       (t - (1.0 + son) * 0.5) / (1.0 - (1.0 + son) * 0.5));
        };
    }

    // overlay: 0 -> dye (warm), 1 -> shock (cool white)
    static Rendering::Colormap overlay_ramp() {
        return [](double t) -> Rendering::Color {
            return t < 0.5 ? Rendering::Color::rgba(255, 224, 138, 255)
                           : Rendering::Color::rgba(245, 250, 255, 255);
        };
    }

    C::Euler2D m_euler{NX, NY, CELL, CELL};
    double m_pback = 1.7;
    Rendering::FieldView m_mach, m_over;
};

} // namespace manifold::Demo
